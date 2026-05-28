#!/usr/bin/env python3
"""
Diagnose which coordinate frame the Pixal3D single-view mesh is in.

Loads an existing init.glb (saved by tto_silhouette.py) and tries several
hypotheses for the world-frame transformation, rendering each against the
GT alpha of the primary view. Reports IoU for each so we can pick the right
one before running the full TTO loop.

Usage:
  python diagnose_frame.py \
    --glb /abs/path/init.glb \
    --transforms_pixel3d /abs/path/transforms_pixel3d.json \
    --primary auto_cam04_angle0 \
    --output /abs/path/diag/
"""
import os, json, math, argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from PIL import Image

PIXAL3D_GRID_ROT = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def opengl_persp(fx, fy, cx, cy, W, H, near, far, device):
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = 1.0 - 2.0 * cx / W
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def render_sil(glctx, verts, faces, T_c2w, fx, fy, cx, cy, W, H):
    device = verts.device
    w2c = torch.linalg.inv(T_c2w.float())
    P = opengl_persp(fx, fy, cx, cy, W, H, 0.01, 100.0, device)
    mvp = P @ w2c
    v_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
    v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)
    pad_w = (8 - W % 8) % 8
    pad_h = (8 - H % 8) % 8
    rast, _ = dr.rasterize(glctx, v_clip, faces, resolution=[H + pad_h, W + pad_w])
    return (rast[0, ..., 3] > 0).float().cpu().numpy()[:H, :W]


def iou(pred, gt):
    p = (pred > 0.5)
    g = (gt > 0.5)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter) / max(int(union), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--transforms_pixel3d', required=True)
    ap.add_argument('--primary', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.transforms_pixel3d) as f:
        T = json.load(f)
    fr_p = next(fr for fr in T['frames'] if fr['file_path'] == f'./images/{args.primary}.png')
    images_dir = T['images_dir']

    g = T['global']
    fx_full, fy_full = float(g['fl_x']), float(g['fl_y'])
    cx_full, cy_full = float(g['cx']), float(g['cy'])
    W_full, H_full = int(g['image_width']), int(g['image_height'])
    scale = 1024.0 / max(W_full, H_full)
    W = int(round(W_full * scale))
    H = int(round(H_full * scale))
    fx = fx_full * scale; fy = fy_full * scale
    cx = cx_full * scale; cy = cy_full * scale

    img = Image.open(os.path.join(images_dir, f'{args.primary}.png')).resize((W, H), Image.LANCZOS)
    rgba = np.array(img.convert('RGBA'))
    if 'A' in img.mode:
        gt = (rgba[:, :, 3] > 127).astype(np.float32)
    else:
        rgb = rgba[:, :, :3].astype(np.float32).sum(axis=-1)
        gt = (rgb > 5).astype(np.float32)
    print(f'[Diag] image: {W}x{H}, GT alpha pixels: {int(gt.sum())}')
    Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(args.output, 'gt_alpha.png'))

    m = trimesh.load(args.glb, force='mesh')
    raw_v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device='cuda')
    faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device='cuda')
    print(f'[Diag] mesh: V={raw_v.shape[0]}, F={faces.shape[0]}, '
          f'raw bbox=[{raw_v.min().item():.3f}, {raw_v.max().item():.3f}]')

    R_grid = PIXAL3D_GRID_ROT.cuda()
    v_canonical = raw_v @ R_grid.T  # standard → canonical-world

    # Pixal3D's inference.py applies this rotation to the GLB before export!
    # So if we load init.glb saved by tto_silhouette.py, the mesh has NOT been
    # through that rotation (we exported via glb.export without applying it
    # because raw glb came from o_voxel.postprocess.to_glb).
    # Actually check: tto_silhouette.py calls glb.export() WITHOUT apply_transform.

    T_primary = torch.tensor(fr_p['transform_matrix_aligned'], dtype=torch.float32, device='cuda')
    distance = float(fr_p['distance'])

    # Build front_view_transform_matrix at this distance.
    T_canonical = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -distance],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    glctx = dr.RasterizeCudaContext(device=torch.device('cuda'))

    def transform(M, v):
        v_h = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
        return (M @ v_h.T).T[:, :3].contiguous()

    hypotheses = {
        'H0_raw_no_rot': raw_v,
        'H1_canonical': v_canonical,
        'H2_T_primary_x_canonical': transform(T_primary, v_canonical),
        'H3_T_primary_x_invTcan_x_canonical': transform(T_primary @ torch.linalg.inv(T_canonical), v_canonical),
        'H4_invTcan_x_canonical': transform(torch.linalg.inv(T_canonical), v_canonical),
        'H5_T_primary_x_raw': transform(T_primary, raw_v),
        'H6_invTprimary_x_canonical': transform(torch.linalg.inv(T_primary), v_canonical),
    }

    print('\n[Diag] Trying hypotheses (primary GT IoU):')
    for name, verts in hypotheses.items():
        sil = render_sil(glctx, verts, faces, T_primary, fx, fy, cx, cy, W, H)
        score = iou(sil, gt)
        print(f'  {name:50s}  IoU={score:.3f}  pred_pix={int(sil.sum())}')
        Image.fromarray((sil * 255).astype(np.uint8)).save(
            os.path.join(args.output, f'pred_{name}.png')
        )
        # color overlay
        ov = np.stack([sil, gt, np.zeros_like(gt)], axis=-1) * 255
        Image.fromarray(ov.astype(np.uint8)).save(
            os.path.join(args.output, f'overlay_{name}_iou{score:.2f}.png')
        )

    # Also try: render H1 with T_canonical as camera (canonical world camera).
    # The plant in the original full image is OFF-CENTER, but in front_view's
    # frame the plant is centered. So this should NOT match the full-image GT.
    sil_can = render_sil(glctx, v_canonical, faces, T_canonical, fx, fy, cx, cy, W, H)
    Image.fromarray((sil_can * 255).astype(np.uint8)).save(
        os.path.join(args.output, 'pred_H1_with_canonical_camera.png')
    )
    print(f'  H1+T_canonical_cam (sanity): IoU vs full-image GT = {iou(sil_can, gt):.3f} '
          '(expected: low; in canonical, plant is centered)')


if __name__ == '__main__':
    main()
