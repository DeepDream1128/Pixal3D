"""
Pixal3D 2-Stage Inference Script
Generate 3D mesh from a single image with MoGe FOV estimation and mesh_scale optimization.
"""

import os
import sys
import argparse
from pathlib import Path

from pixal3dpipeline2stage import Pixal3DPipeline2Stage


def main():
    """Main function - 2-stage inference with MoGe FOV estimation"""
    parser = argparse.ArgumentParser(
        description='Pixal3D 2-Stage Inference - Generate 3D mesh with FOV estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, default='./outputs_2stage',
                        help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Output name (default: image filename)')

    # Model loading
    parser.add_argument('--repo_id', type=str, default="TencentARC/Pixal3D-D",
                        help='HuggingFace repo ID for main models')
    parser.add_argument('--ckpt_dir', type=str, default="./ckpt",
                        help='Local checkpoint directory for main models')
    parser.add_argument('--no_dense_check', action='store_true',
                        help='Disable loading dense_check dit (dense/scale_init)')

    # MoGe & mesh_scale optimization
    parser.add_argument('--no_moge', action='store_true',
                        help='Disable MoGe FOV estimation (use fixed camera_angle_x)')
    parser.add_argument('--camera_angle_x', type=float, default=None,
                        help='Fixed camera FOV angle in radians (only used when --no_moge)')
    parser.add_argument('--mesh_scale', type=float, default=0.5,
                        help='Initial mesh scale')
    parser.add_argument('--no_optimize_mesh_scale', action='store_true',
                        help='Disable iterative mesh_scale optimization')
    parser.add_argument('--target_padding', type=int, default=3,
                        help='Target boundary padding for mesh_scale optimization')
    parser.add_argument('--max_optim_iterations', type=int, default=2,
                        help='Max iterations for mesh_scale optimization')

    # Inference parameters
    parser.add_argument('--dense_steps', type=int, default=50,
                        help='Dense inference steps')
    parser.add_argument('--dense_guidance_scale', type=float, default=7.0,
                        help='Dense guidance scale')
    parser.add_argument('--dense_seed', type=int, default=0,
                        help='Dense random seed')
    parser.add_argument('--sparse_512_steps', type=int, default=30,
                        help='Sparse 512 inference steps')
    parser.add_argument('--sparse_512_guidance_scale', type=float, default=7.0,
                        help='Sparse 512 guidance scale')
    parser.add_argument('--sparse_1024_steps', type=int, default=15,
                        help='Sparse 1024 inference steps')
    parser.add_argument('--sparse_1024_guidance_scale', type=float, default=7.0,
                        help='Sparse 1024 guidance scale')
    parser.add_argument('--sparse_seed', type=int, default=0,
                        help='Sparse random seed')

    # Post-processing
    parser.add_argument('--dense_threshold', type=float, default=0.1,
                        help='Dense decoding threshold')
    parser.add_argument('--mc_threshold', type=float, default=0.2,
                        help='Marching cubes threshold')

    args = parser.parse_args()

    # Check input image
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Setup output directory
    os.makedirs(args.output, exist_ok=True)
    output_name = args.name or Path(args.image).stem
    save_path = os.path.join(args.output, output_name)
    os.makedirs(save_path, exist_ok=True)

    use_moge = not args.no_moge
    optimize_mesh_scale = not args.no_optimize_mesh_scale

    print("=" * 60)
    print("Pixal3D 2-Stage Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Output: {save_path}")
    print(f"MoGe FOV estimation: {use_moge}")
    print(f"Mesh scale optimization: {optimize_mesh_scale}")
    print(f"Dense check model: {'disabled' if args.no_dense_check else 'dense/scale_init'}")
    print("=" * 60)

    # Load model
    print("\n[1/2] Loading model...")
    pipeline = Pixal3DPipeline2Stage.from_pretrained(
        ckpt_dir=args.ckpt_dir,
        repo_id=args.repo_id,
        use_moge=use_moge,
        use_dense_check=not args.no_dense_check,
    )

    # Run inference
    print("\n[2/2] Running 2-stage inference...")

    infer_kwargs = dict(
        dense_steps=args.dense_steps,
        dense_guidance_scale=args.dense_guidance_scale,
        dense_seed=args.dense_seed,
        sparse_512_steps=args.sparse_512_steps,
        sparse_512_guidance_scale=args.sparse_512_guidance_scale,
        sparse_1024_steps=args.sparse_1024_steps,
        sparse_1024_guidance_scale=args.sparse_1024_guidance_scale,
        sparse_seed=args.sparse_seed,
        dense_threshold=args.dense_threshold,
        mc_threshold=args.mc_threshold,
        mesh_scale=args.mesh_scale,
        optimize_mesh_scale=optimize_mesh_scale,
        target_padding=args.target_padding,
        max_optim_iterations=args.max_optim_iterations,
    )

    if not use_moge:
        # When MoGe is disabled, must provide fixed camera_angle_x
        if args.camera_angle_x is None:
            print("Error: --camera_angle_x is required when --no_moge is set")
            sys.exit(1)
        # Fall back to base pipeline infer with fixed params
        from pixal3dpipeline import Pixal3DPipeline
        mesh = Pixal3DPipeline.infer(
            pipeline,
            image=args.image,
            camera_angle_x=args.camera_angle_x,
            mesh_scale=args.mesh_scale,
            dense_steps=args.dense_steps,
            dense_guidance_scale=args.dense_guidance_scale,
            dense_seed=args.dense_seed,
            sparse_512_steps=args.sparse_512_steps,
            sparse_512_guidance_scale=args.sparse_512_guidance_scale,
            sparse_1024_steps=args.sparse_1024_steps,
            sparse_1024_guidance_scale=args.sparse_1024_guidance_scale,
            sparse_seed=args.sparse_seed,
            dense_threshold=args.dense_threshold,
            mc_threshold=args.mc_threshold,
        )
    else:
        mesh = pipeline.infer_from_image(image_path=args.image, **infer_kwargs)

    # Save result
    output_mesh_path = os.path.join(save_path, "mesh.ply")
    mesh.export(output_mesh_path)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)
    print(f"Mesh saved to: {output_mesh_path}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
