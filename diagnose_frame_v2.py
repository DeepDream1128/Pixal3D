#!/usr/bin/env python3
"""
Diagnose mesh frame against PRIMARY's preprocess_image output (1024x1024).

Hypothesis: the mesh is in pre-R_grid canonical frame, [-0.5, 0.5]^3, and
the primary camera in this frame is exactly front_view_transform_matrix
(camera at (0, -distance, 0), looking at origin). Rendering should match
the alpha of preprocess_image(primary) at fov=fov_crop.

If this works, we then render aux views by composing:
  T_aux_in_canonical = T_canonical @ inv(T_primary_aligned) @ T_aux_aligned
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


def opengl_persp(fovx, aspect, near, far, device):
    f = 1.0 / math.tan(fovx / 2.0)
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = f
    P[1, 1] = f / aspect  # square: aspect=1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def render_sil(glctx, verts, faces, T_c2w, fov_x, W, H, apply_R_grid=False):
    device = verts.device
    if apply_R_grid:
        R = PIXAL3D_GRID_ROT.to(device)
        verts = verts @ R.T
    w2c = torch.linalg.inv(T_c2w.float())
    P = opengl_persp(fov_x, W / H, 0.01, 100.0, device)
    mvp = P @ w2c
    v_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
    v_clip = (mvp @ v_h.T).T.contiguous().unsqueeze(0)
    pad = (8 - W % 8) % 8
    rast, _ = dr.rasterize(glctx, v_clip, faces, resolution=[H + pad, W + pad])
    return (rast[0, ..., 3] > 0).float().cpu().numpy()[:H, :W]


def iou(p, g):
    pb = (p > 0.5); gb = (g > 0.5)
    inter = (pb & gb).sum(); union = (pb | gb).sum()
    return float(inter) / max(int(union), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--preprocessed_alpha', required=True,
                    help='Path to preprocess_image output saved during single-view inference (1024x1024).')
    ap.add_argument('--transforms_pixel3d', required=True)
    ap.add_argument('--primary', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.transforms_pixel3d) as f:
        T = json.load(f)
    fr_p = next(fr for fr in T['frames'] if fr['file_path'] == f'./images/{args.primary}.png')

    fl_x_full = float(T['global']['fl_x'])
    bbox = fr_p['alpha_bbox']
    crop_size = float(bbox['size_px']) * float(bbox['padding'])
    fov_crop = 2.0 * math.atan(crop_size / (2.0 * fl_x_full))
    distance = float(fr_p['distance'])
    print(f'[Diag2] fov_crop={math.degrees(fov_crop):.2f}°, distance={distance:.3f}')

    # GT: alpha of preprocess_image output (1024x1024 centered crop, BG=black)
    img = Image.open(args.preprocessed_alpha)
    if img.mode != 'RGBA':
        # bg-removed image saved as RGB with black bg => alpha is "any pixel != 0"
        a_img = np.array(img.convert('RGB')).astype(np.float32).sum(axis=-1)
        gt = (a_img > 5).astype(np.float32)
    else:
        gt = (np.array(img)[:, :, 3] > 127).astype(np.float32)
    H, W = gt.shape
    print(f'[Diag2] preprocess image: {W}x{H}, GT pixels = {int(gt.sum())}')
    Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(args.output, 'gt_alpha_pre.png'))

    m = trimesh.load(args.glb, force='mesh')
    raw_v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device='cuda')
    faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device='cuda')
    print(f'[Diag2] mesh: V={raw_v.shape[0]}, F={faces.shape[0]}, '
          f'bbox=[{raw_v.min().item():.3f}, {raw_v.max().item():.3f}]')

    # canonical primary camera = front_view_transform_matrix(distance)
    T_canonical = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -distance],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    glctx = dr.RasterizeCudaContext(device=torch.device('cuda'))

    # Try each: with/without R_grid applied; with positive/negative distance, etc.
    cases = {
        'A_raw_canonical_camera': (raw_v, T_canonical, False),
        'B_raw_canonical_camera_with_Rgrid': (raw_v, T_canonical, True),
    }

    print('\n[Diag2] Rendering each hypothesis vs preprocess GT:')
    for name, (verts, cam, apply_R) in cases.items():
        sil = render_sil(glctx, verts, faces, cam, fov_crop, W, H, apply_R_grid=apply_R)
        score = iou(sil, gt)
        print(f'  {name:50s}  IoU={score:.3f}  pred={int(sil.sum())}  gt={int(gt.sum())}')
        ov = np.stack([sil, gt, np.zeros_like(gt)], axis=-1) * 255
        Image.fromarray(ov.astype(np.uint8)).save(
            os.path.join(args.output, f'overlay_{name}_iou{score:.2f}.png')
        )
        Image.fromarray((sil * 255).astype(np.uint8)).save(
            os.path.join(args.output, f'pred_{name}.png')
        )


if __name__ == '__main__':
    main()
