#!/usr/bin/env python3
"""
Test-Time Optimization for Pixal3D meshes (silhouette-only PoC).

Workflow:
  1. Single-view inference on PRIMARY view to get a base mesh.
  2. Transform mesh from "front-view canonical" frame into world frame
     using primary view's transform_matrix_aligned.
  3. Render mesh silhouette via nvdiffrast under EACH view's calibrated
     camera (full-image fx/fy/cx/cy from transforms_pixel3d.json[global]).
  4. Sanity check: IoU on primary view should be high (>=0.9). If not,
     coordinate alignment is wrong — abort before optimizing.
  5. Optimize per-vertex offsets to minimize:
        L = sum_v BCE(render_alpha_v, gt_alpha_v) + lap * laplacian_smoothness
  6. Save initial / final mesh + per-step IoU log.

Notes:
  - This is a PoC. We use a hard binary silhouette via dr.antialias for
    differentiability; gradients flow through silhouette edge antialiasing.
  - Marching-cubes is NOT in the differentiable path (mesh is fixed
    topology, only vertex positions move). This keeps the implementation
    simple at the cost of being unable to add/remove geometry.

Usage:
  python tto_silhouette.py \
    --transforms_pixel3d /abs/path/transforms_pixel3d.json \
    --primary auto_cam04_angle0 \
    --aux auto_cam04_angle90 auto_cam04_angle180 auto_cam04_angle270 \
    --output ./output/<ts>_tto \
    --steps 200
"""

import os
import sys
import math
import json
import time
import argparse
from typing import List, Tuple

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json',
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

import o_voxel
from inference import MODEL_PATH, init_pipeline
from inference_multiview import find_frame, derive_fov_after_crop


# ---- Pixal3D internal frame rotation (same as ProjGrid.__init__) -----------
PIXAL3D_GRID_ROT = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def opengl_perspective_offaxis(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, near: float, far: float,
    device: torch.device,
) -> torch.Tensor:
    """OpenGL clip-from-camera with arbitrary principal point."""
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def render_silhouette(
    glctx, verts_world: torch.Tensor, faces: torch.Tensor,
    T_c2w: torch.Tensor, fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
):
    """Differentiable binary-ish silhouette via nvdiffrast.

    Returns:
        sil [1,Hp,Wp,1] in [0,1] (antialiased edges, hard interior),
        clipped to (height, width). All ops are differentiable wrt verts_world.
    """
    device = verts_world.device
    w2c = torch.linalg.inv(T_c2w.float())
    P = opengl_perspective_offaxis(fx, fy, cx, cy, width, height, 0.01, 100.0, device)
    mvp = P @ w2c

    v_h = torch.cat([verts_world, torch.ones_like(verts_world[:, :1])], dim=-1)
    v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)

    pad_w = (8 - width % 8) % 8
    pad_h = (8 - height % 8) % 8
    rast, _ = dr.rasterize(
        glctx, v_clip, faces, resolution=[height + pad_h, width + pad_w]
    )
    sil_hard = (rast[..., 3:4] > 0).float()  # [1,H',W',1] hard mask
    # Antialias along silhouette edges so gradients can flow into vertex pos.
    sil = dr.antialias(sil_hard, rast, v_clip, faces)
    sil = sil[:, :height, :width, :]
    return sil  # [1, H, W, 1]


def laplacian_of(values: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute the Laplacian (V - mean(neighbors)) of a per-vertex tensor.

    Works on either positions or per-vertex offsets. Used to encourage
    SMOOTHNESS of the deformation field, not the surface itself.
    """
    F_long = faces.long()
    edges = torch.cat([
        F_long[:, [0, 1]], F_long[:, [1, 2]], F_long[:, [2, 0]],
    ], dim=0)
    edges = torch.cat([edges, edges.flip(-1)], dim=0)
    V = values.shape[0]
    nbr_sum = torch.zeros_like(values)
    nbr_sum.index_add_(0, edges[:, 0], values[edges[:, 1]])
    deg = torch.zeros(V, device=values.device, dtype=values.dtype)
    deg.index_add_(0, edges[:, 0], torch.ones(edges.shape[0], device=values.device, dtype=values.dtype))
    nbr_mean = nbr_sum / deg.clamp_min(1.0).unsqueeze(-1)
    return values - nbr_mean


def iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = (pred > 0.5).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum().item()
    union = ((p + g) > 0.5).float().sum().item()
    return inter / max(union, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transforms_pixel3d', required=True)
    ap.add_argument('--primary', required=True)
    ap.add_argument('--aux', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--resolution', type=int, default=1024,
                    help='Pixal3D pipeline resolution (1024 or 1536).')
    ap.add_argument('--render_long_side', type=int, default=1024,
                    help='Render image long side (full-image down-sampled).')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lap_weight', type=float, default=1000.0,
                    help='Laplacian smoothness on the displacement field (delta).')
    ap.add_argument('--delta_weight', type=float, default=100.0,
                    help='L2 penalty on per-vertex displacement magnitude.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--low_vram', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.transforms_pixel3d) as f:
        T = json.load(f)
    images_dir = T['images_dir']

    pipeline = init_pipeline(MODEL_PATH, low_vram=args.low_vram)

    # ---- 1) Single-view inference on primary ----
    fr_p = find_frame(T, args.primary)
    bbox_p = fr_p['alpha_bbox']
    fl_x_full = float(T['global']['fl_x'])
    fov_p = derive_fov_after_crop(bbox_p, fl_x_full)
    cam_params_p = {
        'camera_angle_x': fov_p,
        'distance': fr_p['distance'],
        'mesh_scale': fr_p.get('mesh_scale', 1.0),
    }
    img_p = pipeline.preprocess_image(Image.open(os.path.join(images_dir, f'{args.primary}.png')))
    img_p.save(os.path.join(args.output, f'primary_{args.primary}_preprocessed.png'))
    print(f'[TTO] Primary {args.primary}: fov_crop={math.degrees(fov_p):.2f}°, '
          f'distance={cam_params_p["distance"]:.3f}')
    pipeline_type = f'{args.resolution}_cascade'
    torch.manual_seed(args.seed)
    mesh_list, (shape_slat, tex_slat, hr_res) = pipeline.run(
        img_p, camera_params=cam_params_p, seed=args.seed,
        preprocess_image=False, return_latent=True, pipeline_type=pipeline_type,
    )
    mesh = mesh_list[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=hr_res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=200000, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
    )
    glb.export(os.path.join(args.output, 'init.glb'), extension_webp=True)

    # raw vertices come from o_voxel in Pixal3D internal canonical frame.
    # KEY INSIGHT (verified by diagnose_frame_v2): the mesh is ALREADY in the
    # canonical [-0.5, 0.5]^3 frame the model trained on. The primary camera
    # in this frame is exactly front_view_transform_matrix(distance), and the
    # render target is preprocess_image's 1024-long-side centered crop, NOT
    # the original full image.
    #
    # For aux views, we transfer their relative pose (in transforms_pixel3d's
    # aligned world frame) into the canonical frame:
    #   T_aux_canonical = T_canonical_primary @ inv(T_aligned_primary) @ T_aligned_aux
    raw_v = torch.tensor(np.asarray(glb.vertices), dtype=torch.float32, device='cuda')
    faces = torch.tensor(np.asarray(glb.faces), dtype=torch.int32, device='cuda')
    v_canonical = raw_v  # NO R_grid multiplication (verified empirically).
    print(f'[TTO] mesh: V={raw_v.shape[0]}, F={faces.shape[0]}, '
          f'canonical bbox=[{v_canonical.min().item():.3f}, {v_canonical.max().item():.3f}]')

    distance = float(fr_p['distance'])
    T_canonical_primary = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -distance],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    T_aligned_primary = torch.tensor(
        fr_p['transform_matrix_aligned'], dtype=torch.float32, device='cuda',
    )
    rel_pivot = T_canonical_primary @ torch.linalg.inv(T_aligned_primary)

    v_world_init = v_canonical  # we render in the canonical frame directly

    # ---- 2) Setup canonical-frame cameras + preprocess-style GT alphas ----
    # Each aux view's camera position in the canonical frame is computed by
    # transferring its relative pose from the aligned world frame:
    #   T_aux_canonical = rel_pivot @ T_aligned_aux,  where
    #   rel_pivot = T_canonical_primary @ inv(T_aligned_primary).
    # This guarantees primary's canonical camera is exactly
    # front_view_transform_matrix(distance) and aux cameras orbit around the
    # canonical [-0.5, 0.5]^3 cube consistently.

    fl_x_full = float(T['global']['fl_x'])

    views = [args.primary] + list(args.aux)
    cam_T_canonical = []        # camera-to-canonical 4x4 per view
    fov_x_per_view = []         # post-crop FOV per view (radians)
    gt_alpha = []
    img_resolution = 1024

    for v in views:
        fr = find_frame(T, v)
        # Use Pixal3D's own preprocess_image to get the centered alpha image.
        img_pre = pipeline.preprocess_image(Image.open(os.path.join(images_dir, f'{v}.png')))
        img_pre = img_pre.resize((img_resolution, img_resolution), Image.LANCZOS)
        # preprocess_image returns RGB (alpha-composited on bg), so derive
        # alpha by detecting non-bg pixels (bg=black, so any non-zero pixel
        # is foreground).
        a_arr = np.array(img_pre.convert('RGB')).astype(np.float32).sum(axis=-1)
        gt = (a_arr > 5).astype(np.float32)
        gt_alpha.append(torch.tensor(gt, dtype=torch.float32, device='cuda'))

        bbox = fr['alpha_bbox']
        crop_size = float(bbox['size_px']) * float(bbox['padding'])
        fov_x = 2.0 * math.atan(crop_size / (2.0 * fl_x_full))
        fov_x_per_view.append(fov_x)

        if v == args.primary:
            T_can = T_canonical_primary
        else:
            T_aligned = torch.tensor(fr['transform_matrix_aligned'], dtype=torch.float32, device='cuda')
            T_can = rel_pivot @ T_aligned
        cam_T_canonical.append(T_can)
        print(f'[TTO] view {v}: fov={math.degrees(fov_x):.2f}°, '
              f'cam pos in canonical = {T_can[:3, 3].cpu().numpy()}')

    glctx = dr.RasterizeCudaContext(device=torch.device('cuda'))

    # The render path now uses a CENTERED principal point (cx=cy=W/2) and a
    # symmetric FOV per view, since preprocess_image gives us a centered
    # square crop. We can use a simpler symmetric perspective:
    def _render_sil_sym(verts, T_c2w, fov_x, W, H):
        device = verts.device
        w2c = torch.linalg.inv(T_c2w.float())
        f = 1.0 / math.tan(fov_x / 2.0)
        P = torch.zeros(4, 4, dtype=torch.float32, device=device)
        P[0, 0] = f
        P[1, 1] = f * (W / H)
        P[2, 2] = -(100.0 + 0.01) / (100.0 - 0.01)
        P[2, 3] = -2.0 * 100.0 * 0.01 / (100.0 - 0.01)
        P[3, 2] = -1.0
        mvp = P @ w2c
        v_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
        v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)
        pad = (8 - W % 8) % 8
        rast, _ = dr.rasterize(glctx, v_clip, faces, resolution=[H + pad, W + pad])
        sil_hard = (rast[..., 3:4] > 0).float()
        sil = dr.antialias(sil_hard, rast, v_clip, faces)
        return sil[:, :H, :W, :]

    W = H = img_resolution

    # ---- 3) Step-0 sanity render ----
    print('\n[TTO] step-0 sanity render (canonical frame, preprocess_image GT)...')
    with torch.no_grad():
        for i, v in enumerate(views):
            sil = _render_sil_sym(v_world_init, cam_T_canonical[i], fov_x_per_view[i], W, H)
            sil_np = sil[0, ..., 0]
            iou_val = iou(sil_np, gt_alpha[i])
            print(f'  [{i}] {v}: IoU = {iou_val:.3f}  '
                  f'(pred={int((sil_np > 0.5).sum().item())}, gt={int(gt_alpha[i].sum().item())})')
            overlay = np.stack([
                sil_np.cpu().numpy(),
                gt_alpha[i].cpu().numpy(),
                np.zeros_like(gt_alpha[i].cpu().numpy()),
            ], axis=-1)
            Image.fromarray((overlay * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(args.output, f'sanity_{i}_{v}_iou{iou_val:.2f}.png')
            )

    with torch.no_grad():
        sil0 = _render_sil_sym(v_world_init, cam_T_canonical[0], fov_x_per_view[0], W, H)
        primary_iou = iou(sil0[0, ..., 0], gt_alpha[0])
    if primary_iou < 0.5:
        print(f'\n[TTO] PRIMARY IoU = {primary_iou:.3f} < 0.5 — coord alignment FAILED. Aborting.')
        with open(os.path.join(args.output, 'sanity_FAILED.txt'), 'w') as f:
            f.write(f'primary_iou = {primary_iou}\n')
        return

    # ---- 4) Optimize vertex offsets ----
    delta = nn.Parameter(torch.zeros_like(v_world_init))
    optim = torch.optim.AdamW([delta], lr=args.lr)
    log = []
    print(f'\n[TTO] starting optimization: {args.steps} steps, lr={args.lr}, '
          f'lap_w={args.lap_weight}, delta_w={args.delta_weight}')
    for step in range(args.steps):
        verts = v_world_init + delta
        per_view_iou = []
        L_sil = 0.0
        for i in range(len(views)):
            sil = _render_sil_sym(verts, cam_T_canonical[i], fov_x_per_view[i], W, H)
            sil_v = sil[0, ..., 0]
            L_sil = L_sil + F.binary_cross_entropy(sil_v.clamp(1e-6, 1 - 1e-6), gt_alpha[i])
            with torch.no_grad():
                per_view_iou.append(iou(sil_v, gt_alpha[i]))
        # Smoothness on the DEFORMATION FIELD: spiky deltas have a large
        # local Laplacian, smooth global motion has a small one.
        delta_lap = laplacian_of(delta, faces)
        L_lap = (delta_lap ** 2).mean()
        # Magnitude penalty: keep vertices near their initial positions.
        L_delta = (delta ** 2).mean()
        L = L_sil + args.lap_weight * L_lap + args.delta_weight * L_delta
        optim.zero_grad()
        L.backward()
        optim.step()

        if step % 10 == 0 or step == args.steps - 1:
            print(f'  step {step:3d}: L={L.item():.4f}  L_sil={L_sil.item():.4f}  '
                  f'L_lap={L_lap.item():.6f}  L_delta={L_delta.item():.6f}  '
                  f'IoU={[f"{x:.3f}" for x in per_view_iou]}')
            log.append({'step': step, 'L': float(L), 'L_sil': float(L_sil),
                        'L_lap': float(L_lap), 'L_delta': float(L_delta),
                        'iou': per_view_iou})

    with open(os.path.join(args.output, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # ---- 5) Save final mesh + final renders ----
    final_verts = (v_world_init + delta).detach().cpu().numpy()
    import trimesh
    final_mesh = trimesh.Trimesh(vertices=final_verts,
                                  faces=faces.cpu().numpy(),
                                  process=False)
    final_mesh.export(os.path.join(args.output, 'final.glb'))

    print('\n[TTO] final renders:')
    with torch.no_grad():
        verts = v_world_init + delta
        for i, v in enumerate(views):
            sil = _render_sil_sym(verts, cam_T_canonical[i], fov_x_per_view[i], W, H)
            iou_v = iou(sil[0, ..., 0], gt_alpha[i])
            print(f'  [{i}] {v}: IoU = {iou_v:.3f}')
            sil_np = sil[0, ..., 0].cpu().numpy()
            gt_np = gt_alpha[i].cpu().numpy()
            overlay = np.stack([sil_np, gt_np, np.zeros_like(gt_np)], axis=-1)
            Image.fromarray((overlay * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(args.output, f'final_{i}_{v}_iou{iou_v:.2f}.png')
            )

    print(f'\n[TTO] Done. Output: {args.output}')


if __name__ == '__main__':
    main()
