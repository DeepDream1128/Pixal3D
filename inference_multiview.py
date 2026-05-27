#!/usr/bin/env python3
"""
Pixal3D Multi-View Inference (research / scheme A: inference-time aggregation)

Usage:
  python inference_multiview.py \\
    --transforms_pixel3d /abs/path/transforms_pixel3d.json \\
    --views auto_cam04_angle0 auto_cam04_angle90 auto_cam04_angle180 auto_cam04_angle270 \\
    --output ./run/cam04_4view_1024.glb \\
    --resolution 1024

Each view name resolves to ./images/<name>.png via transforms.json's file_path.
"""

import os
import argparse
import json
import math
import time
from typing import List

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json',
)

import numpy as np
import torch
from PIL import Image

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# Reuse helpers from inference.py
from inference import (
    MODEL_PATH,
    IMAGE_COND_CONFIGS,
    build_image_cond_model,
    init_pipeline,
)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def find_frame(transforms: dict, view_name: str) -> dict:
    target = f'./images/{view_name}.png'
    for fr in transforms['frames']:
        if fr['file_path'] == target:
            return fr
    raise KeyError(f"View {view_name!r} not found in transforms_pixel3d.json (looked for {target}).")


def derive_fov_after_crop(bbox: dict, fl_x: float, padding_override: float = None) -> float:
    """preprocess_image 会用 bbox * 1.1 作为方裁剪边长。"""
    pad = padding_override if padding_override is not None else bbox['padding']
    crop_size = float(bbox['size_px']) * pad
    return 2.0 * math.atan(crop_size / (2.0 * fl_x))


def preprocess_for_pipeline(pipeline, image_path: str) -> Image.Image:
    img = Image.open(image_path)
    return pipeline.preprocess_image(img)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_multiview(
    transforms_pixel3d_path: str,
    view_names: List[str],
    output_path: str,
    seed: int = 42,
    resolution: int = 1024,
    low_vram: bool = False,
    global_strategy: str = 'primary',
    images_dir_override: str = None,
):
    with open(transforms_pixel3d_path, 'r') as f:
        T = json.load(f)
    fl_x = float(T['global']['fl_x'])
    images_dir = images_dir_override or T.get('images_dir')

    pipeline = init_pipeline(MODEL_PATH, low_vram=low_vram)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]

    images, camera_params_list = [], []
    print(f'\n[MV] Loading {len(view_names)} view(s):')
    for i, name in enumerate(view_names):
        fr = find_frame(T, name)
        bbox = fr.get('alpha_bbox')
        if bbox is None:
            raise RuntimeError(f'{name}: alpha_bbox missing in transforms_pixel3d.json; rerun calibrate_transforms.')
        img_path = os.path.join(images_dir, f'{name}.png')
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        pre = preprocess_for_pipeline(pipeline, img_path)
        pre.save(os.path.join(out_dir, f'{base}_view{i:02d}_{name}_preprocessed.png'))
        fov_crop = derive_fov_after_crop(bbox, fl_x)
        cp = {
            'camera_angle_x': fov_crop,
            'distance': fr['distance'],
            'mesh_scale': fr.get('mesh_scale', 1.0),
            'transform_matrix': fr['transform_matrix_aligned'],
        }
        images.append(pre)
        camera_params_list.append(cp)
        print(f'  view{i:02d} {name}: fov_crop={math.degrees(fov_crop):.2f}°, '
              f'dist={cp["distance"]:.3f}, ms={cp["mesh_scale"]:.3f}')

    pipeline_type = f'{resolution}_cascade'
    print(f'\n[MV] Running run_multiview (pipeline_type={pipeline_type}, '
          f'global_strategy={global_strategy})...')
    torch.manual_seed(seed)
    t0 = time.time()
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run_multiview(
        images=images,
        camera_params_list=camera_params_list,
        seed=seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        global_strategy=global_strategy,
    )
    print(f'[MV] run_multiview done in {time.time() - t0:.1f}s, res={res}')

    mesh = mesh_list[0]
    print('[MV] Extracting GLB...')
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    rot = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    glb.export(output_path, extension_webp=True)
    print(f'[MV] GLB saved to: {output_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--transforms_pixel3d', required=True)
    ap.add_argument('--views', nargs='+', required=True,
                    help='View names, e.g. auto_cam04_angle0 auto_cam04_angle90 ...')
    ap.add_argument('--output', required=True)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--low_vram', action='store_true')
    ap.add_argument('--global_strategy', choices=['primary', 'mean'], default='primary')
    ap.add_argument('--images_dir', default=None,
                    help='Override images directory. Default uses images_dir from transforms_pixel3d.json.')
    args = ap.parse_args()

    run_multiview(
        transforms_pixel3d_path=args.transforms_pixel3d,
        view_names=args.views,
        output_path=args.output,
        seed=args.seed,
        resolution=args.resolution,
        low_vram=args.low_vram,
        global_strategy=args.global_strategy,
        images_dir_override=args.images_dir,
    )
