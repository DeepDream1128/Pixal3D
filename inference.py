"""
Pixal3D Inference Script
Generate 3D mesh from a single image
"""

import os
import sys
import argparse
from pathlib import Path

from pixal3dpipeline import Pixal3DPipeline


def main():
    """Main function - single image inference"""
    parser = argparse.ArgumentParser(
        description='Pixal3D Inference - Generate 3D mesh from single image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Output name (default: image filename)')
    
    # Model loading
    parser.add_argument('--repo_id', type=str, default="TencentARC/Pixal3D-D",
                        help='HuggingFace repo ID (default: TencentARC/Pixal3D-D)')
    
    # Camera parameters
    parser.add_argument('--camera_angle_x', type=float, default=0.2,
                        help='Camera FOV angle in radians')
    parser.add_argument('--mesh_scale', type=float, default=0.9,
                        help='Mesh scale')
    
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
    
    print("=" * 60)
    print("🚀 Pixal3D Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Output: {save_path}")
    print("=" * 60)
    
    # Load model and run inference
    print("\n[1/2] Loading model...")
    pipeline = Pixal3DPipeline.from_pretrained(repo_id=args.repo_id)
    
    print("\n[2/2] Running inference...")
    mesh = pipeline.infer_from_image(
        image_path=args.image,
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
    
    # Save result
    output_mesh_path = os.path.join(save_path, "mesh.ply")
    mesh.export(output_mesh_path)
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print("=" * 60)
    print(f"Mesh saved to: {output_mesh_path}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
