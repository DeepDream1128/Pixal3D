#!/usr/bin/env python3
"""
Step 0 sanity check (mesh-silhouette version) for "single-view + multi-view TTO" plan.

Goal: render the mesh produced by single-view inference on the PRIMARY view
into each AUX view's calibrated camera, and measure how well the mesh
silhouette covers the GT alpha mask of that aux view. This reflects the
actual 3D geometry coverage (not just sparse voxel centers), which is the
signal we'd be optimizing against in test-time optimization.

Pipeline:
  1) Run pipeline.run() on the primary view (calibrated camera params),
     get mesh.vertices / mesh.faces and HR sparse voxel coords (for stats).
  2) Apply Pixal3D's internal rotation to align mesh verts with the
     transform_matrix_aligned camera frame (world frame).
  3) For each AUX view: render binary silhouette via nvdiffrast at 1024x1024,
     compare with GT alpha. Report recall / precision / IoU.
  4) Save side-by-side overlays + JSON report.

Usage:
  python check_voxel_coverage.py \
    --transforms_pixel3d /abs/path/transforms_pixel3d.json \
    --primary auto_cam04_angle0 \
    --aux auto_cam04_angle90 auto_cam04_angle180 auto_cam04_angle270 \
    --output ./output/voxel_coverage_<ts> \
    --resolution 1024
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
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

import o_voxel
from inference import MODEL_PATH, init_pipeline
from inference_multiview import find_frame, derive_fov_after_crop


# ----------------------------------------------------------------------------
# Coordinate / projection helpers
# ----------------------------------------------------------------------------

# Pixal3D internal frame: a fixed rotation is applied to grid_points before
# back-projection (see ProjGrid.__init__). Mesh vertices live in the
# pre-rotation canonical frame, so we apply this rotation to bring them into
# the world frame that ``transform_matrix_aligned`` operates in.
PIXAL3D_GRID_ROT = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def to_world_frame(pts: torch.Tensor) -> torch.Tensor:
    """Apply ProjGrid's rotation to bring canonical-frame points to world frame."""
    R = PIXAL3D_GRID_ROT.to(pts.device, pts.dtype)
    return pts @ R.T


def opengl_perspective(fovy_rad: float, aspect: float, near: float, far: float,
                       device: torch.device) -> torch.Tensor:
    f = 1.0 / math.tan(fovy_rad / 2.0)
    return torch.tensor([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=torch.float32, device=device)


def opengl_perspective_offaxis(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, near: float, far: float,
    device: torch.device,
) -> torch.Tensor:
    """OpenGL perspective with arbitrary principal point (off-axis frustum).

    Builds an OpenGL clip-from-camera matrix from pinhole intrinsics
    (fx, fy, cx, cy) at image size (width, height). Camera looks down -Z
    (Blender / OpenGL convention).
    """
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def render_mesh_silhouette_full(
    glctx,
    verts_world: torch.Tensor,        # [V,3] cuda, in Pixal3D world frame
    faces: torch.Tensor,              # [F,3] int32 cuda
    transform_matrix: torch.Tensor,   # [4,4] c2w in Pixal3D world frame
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> np.ndarray:
    """Rasterize binary silhouette using FULL pinhole intrinsics. Returns [H, W] uint8."""
    device = verts_world.device
    T = transform_matrix.to(device).float()
    w2c = torch.linalg.inv(T)
    P = opengl_perspective_offaxis(fx, fy, cx, cy, width, height, 0.01, 100.0, device)
    mvp = P @ w2c

    v_h = torch.cat([verts_world, torch.ones_like(verts_world[:, :1])], dim=-1)
    v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)

    # nvdiffrast requires resolution to be a multiple of 8.
    pad_w = (8 - width % 8) % 8
    pad_h = (8 - height % 8) % 8
    rast, _ = dr.rasterize(glctx, v_clip, faces, resolution=[height + pad_h, width + pad_w])
    mask_full = (rast[0, ..., 3] > 0).to(torch.uint8).cpu().numpy()
    return mask_full[:height, :width]


def render_mesh_silhouette(
    glctx,
    verts_world: torch.Tensor,        # [V,3] cuda, in Pixal3D world frame
    faces: torch.Tensor,              # [F,3] int32 cuda
    transform_matrix: torch.Tensor,   # [4,4] c2w in Pixal3D world frame
    camera_angle_x: float,
    resolution: int,
) -> np.ndarray:
    """Rasterize binary silhouette via nvdiffrast. Returns [H, W] uint8 mask."""
    device = verts_world.device
    T = transform_matrix.to(device).float()
    w2c = torch.linalg.inv(T)
    P = opengl_perspective(camera_angle_x, 1.0, 0.01, 100.0, device)
    mvp = P @ w2c                                                                # [4,4]

    v_h = torch.cat([verts_world, torch.ones_like(verts_world[:, :1])], dim=-1)  # [V,4]
    v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)                           # [1,V,4]

    rast, _ = dr.rasterize(glctx, v_clip, faces, resolution=[resolution, resolution])
    mask = (rast[0, ..., 3] > 0).to(torch.uint8).cpu().numpy()                    # [H,W]
    return mask


# ----------------------------------------------------------------------------
# Coverage metrics
# ----------------------------------------------------------------------------

def compute_coverage(pred_mask: np.ndarray, gt_alpha: np.ndarray) -> dict:
    gt_bin = (gt_alpha > 127).astype(np.uint8)
    inter = ((pred_mask > 0) & (gt_bin > 0)).sum()
    pred_n = (pred_mask > 0).sum()
    gt_n = gt_bin.sum()
    union = pred_n + gt_n - inter
    return {
        'recall': float(inter) / max(int(gt_n), 1),
        'precision': float(inter) / max(int(pred_n), 1),
        'iou': float(inter) / max(int(union), 1),
        'pred_pixels': int(pred_n),
        'gt_pixels': int(gt_n),
    }


def overlay_compare(rgba_image: Image.Image, pred_mask: np.ndarray,
                    save_path: str, title: str = ''):
    """Save a side-by-side: [GT alpha | pred voxel mask | overlay] for visual check."""
    img = rgba_image.convert('RGBA').resize(
        (pred_mask.shape[1], pred_mask.shape[0]), Image.BILINEAR
    )
    arr = np.array(img)
    gt_alpha = arr[:, :, 3:4]
    H, W = pred_mask.shape
    pred_vis = (pred_mask * 255).astype(np.uint8)

    panel = np.zeros((H, W * 3 + 2 * 8, 3), dtype=np.uint8)
    gap = 8
    panel[:, :W, :] = np.repeat(gt_alpha, 3, axis=2)
    panel[:, W + gap:2 * W + gap, :] = np.stack([pred_vis] * 3, axis=2)
    overlay = arr[:, :, :3].copy()
    pmask3 = (pred_mask > 0)
    overlay[pmask3] = (overlay[pmask3] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    panel[:, 2 * (W + gap):2 * (W + gap) + W, :] = overlay
    out = Image.fromarray(panel)
    if title:
        d = ImageDraw.Draw(out)
        d.text((10, 10), title, fill=(255, 255, 255))
    out.save(save_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_check(
    transforms_pixel3d_path: str,
    primary_view: str,
    aux_views: List[str],
    output_dir: str,
    resolution: int = 1024,
    seed: int = 42,
    low_vram: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    with open(transforms_pixel3d_path, 'r') as f:
        T = json.load(f)
    fl_x = float(T['global']['fl_x'])
    images_dir = T['images_dir']

    pipeline = init_pipeline(MODEL_PATH, low_vram=low_vram)

    # ---- 1) Single-view inference on primary ----
    fr_p = find_frame(T, primary_view)
    bbox_p = fr_p['alpha_bbox']
    img_p_path = os.path.join(images_dir, f'{primary_view}.png')
    img_p = pipeline.preprocess_image(Image.open(img_p_path))
    img_p.save(os.path.join(output_dir, f'primary_{primary_view}_preprocessed.png'))
    fov_p = derive_fov_after_crop(bbox_p, fl_x)
    cam_params_p = {
        'camera_angle_x': fov_p,
        'distance': fr_p['distance'],
        'mesh_scale': fr_p.get('mesh_scale', 1.0),
    }
    print(f'[Coverage] Primary view {primary_view}: '
          f'fov_crop={math.degrees(fov_p):.2f}°, dist={cam_params_p["distance"]:.3f}')

    pipeline_type = f'{resolution}_cascade'
    print(f'[Coverage] Running single-view ({pipeline_type})...')
    torch.manual_seed(seed)
    t0 = time.time()
    mesh_list, (shape_slat, tex_slat, hr_res) = pipeline.run(
        img_p,
        camera_params=cam_params_p,
        seed=seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
    )
    print(f'[Coverage] single-view done in {time.time() - t0:.1f}s, '
          f'hr_res={hr_res}, voxels={shape_slat.coords.shape[0]}')

    # ---- 1b) Decode to mesh + extract verts/faces ----
    mesh = mesh_list[0]
    print('[Coverage] Building GLB to extract clean verts/faces...')
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=hr_res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
    )
    glb.export(os.path.join(output_dir, f'primary_{primary_view}.glb'),
               extension_webp=True)

    # NOTE: inference.py applies a final 4x4 rotation right before exporting
    # (rot[-1,0,0,0; 0,0,-1,0; 0,-1,0,0; 0,0,0,1]). We do NOT apply it here:
    # we want vertices in the Pixal3D internal canonical frame (the same frame
    # the ProjGrid voxel cube lives in), then we apply PIXAL3D_GRID_ROT to
    # bring them to the same world frame as transform_matrix_aligned.
    raw_verts = torch.tensor(np.asarray(glb.vertices), dtype=torch.float32, device='cuda')
    faces = torch.tensor(np.asarray(glb.faces), dtype=torch.int32, device='cuda')
    verts_world = to_world_frame(raw_verts)
    print(f'[Coverage] mesh: V={raw_verts.shape[0]}, F={faces.shape[0]}, '
          f'world bbox=[{verts_world.min().item():.3f}, {verts_world.max().item():.3f}]')

    glctx = dr.RasterizeCudaContext(device=torch.device('cuda'))

    # ---- Read full-image intrinsics (we render against the ORIGINAL full image,
    # not preprocess_image's crop, so the camera frustum exactly matches
    # transform_matrix_aligned which was calibrated in the original frame).
    g = T['global']
    fx_full = float(g['fl_x'])
    fy_full = float(g['fl_y'])
    cx_full = float(g['cx'])
    cy_full = float(g['cy'])
    W_full = int(g['image_width'])
    H_full = int(g['image_height'])
    print(f'[Coverage] full-image intrinsics: fx={fx_full:.1f}, fy={fy_full:.1f}, '
          f'cx={cx_full:.1f}, cy={cy_full:.1f}, size={W_full}x{H_full}')

    coverage_report = {
        'primary_view': primary_view,
        'pipeline_type': pipeline_type,
        'hr_resolution': int(hr_res),
        'num_voxels': int(shape_slat.coords.shape[0]),
        'mesh_verts': int(raw_verts.shape[0]),
        'mesh_faces': int(faces.shape[0]),
        'full_image_size': [W_full, H_full],
        'aux_views': {},
    }

    # Optional downscale to keep rendering / IO reasonable.
    eval_scale = 1024.0 / max(W_full, H_full)
    eval_W = int(round(W_full * eval_scale))
    eval_H = int(round(H_full * eval_scale))
    fx_eval = fx_full * eval_scale
    fy_eval = fy_full * eval_scale
    cx_eval = cx_full * eval_scale
    cy_eval = cy_full * eval_scale
    print(f'[Coverage] eval render size: {eval_W}x{eval_H} (scale={eval_scale:.4f})')

    aux_with_primary = [primary_view] + list(aux_views)

    for v_name in aux_with_primary:
        fr = find_frame(T, v_name)
        kind = 'primary' if v_name == primary_view else 'aux'

        # GT alpha from the ORIGINAL image (not preprocess_image's crop).
        img_v_path = os.path.join(images_dir, f'{v_name}.png')
        img_orig = Image.open(img_v_path)
        img_eval = img_orig.resize((eval_W, eval_H), Image.LANCZOS)
        rgba = np.array(img_eval.convert('RGBA'))
        if img_orig.mode == 'RGBA':
            gt_alpha = rgba[:, :, 3]
        else:
            rgb_sum = rgba[:, :, :3].astype(np.float32).sum(axis=-1)
            gt_alpha = (rgb_sum > 5).astype(np.uint8) * 255

        T_mat = torch.tensor(fr['transform_matrix_aligned'], dtype=torch.float32)
        pred_mask = render_mesh_silhouette_full(
            glctx, verts_world, faces, T_mat,
            fx_eval, fy_eval, cx_eval, cy_eval, eval_W, eval_H,
        )

        m = compute_coverage(pred_mask, gt_alpha)
        coverage_report['aux_views'][v_name] = {
            'role': kind,
            **m,
        }
        print(f'[Coverage] [{kind}] {v_name}: '
              f'recall={m["recall"]:.3f} precision={m["precision"]:.3f} iou={m["iou"]:.3f} '
              f'(pred_pixels={m["pred_pixels"]}, gt_pixels={m["gt_pixels"]})')

        overlay_compare(
            img_eval, pred_mask,
            os.path.join(output_dir, f'overlay_{kind}_{v_name}.png'),
            title=f'[{kind}] {v_name}  recall={m["recall"]:.2f} iou={m["iou"]:.2f}',
        )

    report_path = os.path.join(output_dir, 'coverage_report.json')
    with open(report_path, 'w') as f:
        json.dump(coverage_report, f, indent=2)
    print(f'\n[Coverage] Report written to {report_path}')

    # ---- 3) Verdict ----
    recalls = [v['recall'] for v in coverage_report['aux_views'].values()]
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    print(f'[Coverage] avg recall over aux views = {avg_recall:.3f}')
    if avg_recall >= 0.7:
        verdict = 'PASS — voxels cover most of the aux silhouettes; latent TTO is feasible.'
    elif avg_recall >= 0.4:
        verdict = 'MARGINAL — partial coverage; latent TTO may help but coords likely also need optimization.'
    else:
        verdict = 'FAIL — single-view prior did not generate voxels in the occluded regions; latent TTO will not work without coord-level optimization or seeding from multi-view stage 1.'
    print(f'[Coverage] Verdict: {verdict}')
    coverage_report['verdict'] = verdict
    coverage_report['avg_recall'] = avg_recall
    with open(report_path, 'w') as f:
        json.dump(coverage_report, f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--transforms_pixel3d', required=True)
    ap.add_argument('--primary', required=True, help='Primary view name (drives single-view).')
    ap.add_argument('--aux', nargs='+', required=True, help='Auxiliary view names for coverage check.')
    ap.add_argument('--output', required=True, help='Output directory for masks/report.')
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--low_vram', action='store_true')
    args = ap.parse_args()

    run_check(
        transforms_pixel3d_path=args.transforms_pixel3d,
        primary_view=args.primary,
        aux_views=args.aux,
        output_dir=args.output,
        resolution=args.resolution,
        seed=args.seed,
        low_vram=args.low_vram,
    )
