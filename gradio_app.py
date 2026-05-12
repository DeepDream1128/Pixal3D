"""
Pixal3D Gradio App
Upload an image and generate a 3D mesh. Supports both automatic (MoGe) and fixed camera parameters.
"""

import os
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"
import torch
import tempfile
import numpy as np
from PIL import Image
from torchvision import transforms

import gradio as gr

from pixal3dpipeline2stage import Pixal3DPipeline2Stage
from pixal3dpipeline import Pixal3DPipeline

# Global pipeline reference
pipeline = None
rmbg = None


def load_pipeline(ckpt_dir="./ckpt", repo_id="TencentARC/Pixal3D-D"):
    """Load all weights at startup."""
    global pipeline, rmbg
    print("Loading Pixal3D 2-Stage pipeline (with MoGe + dense_check)...")
    pipeline = Pixal3DPipeline2Stage.from_pretrained(
        ckpt_dir=ckpt_dir,
        repo_id=repo_id,
        use_moge=True,
        use_dense_check=True,
    )
    print("Pipeline loaded!")
    print("Loading BiRefNet for background removal...")
    from transformers import AutoModelForImageSegmentation
    birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True,
    ).to("cuda:0")
    birefnet_model.eval()
    rmbg = birefnet_model
    print("BiRefNet loaded!")


def remove_background(image_np):
    """Use BiRefNet to remove background and add alpha channel.
    Input: numpy array (H, W, 3) RGB
    Output: numpy array (H, W, 4) RGBA
    """
    pil_img = Image.fromarray(image_np[:, :, :3]).convert('RGB')
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform_image(pil_img).unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        preds = rmbg(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(pil_img.size)
    mask = np.array(mask)
    rgba = np.concatenate([np.array(pil_img), mask[..., None]], axis=-1)
    return rgba


def preprocess_image(image, use_rmbg):
    """Step 1: process image (background removal or use original), return immediately.
    
    use_rmbg=True: run BiRefNet to remove background and generate RGBA
    use_rmbg=False: directly use the original image (RGB or RGBA), skip background removal
    """
    if image is None:
        return None

    if use_rmbg:
        # Run background removal
        if rmbg is None:
            gr.Warning("Background removal model not loaded.")
            return None
        processed = remove_background(image)
    else:
        # Directly use original image, no background removal
        processed = image

    os.makedirs("./gradio_outputs", exist_ok=True)
    Image.fromarray(processed).save("./gradio_outputs/processed.png")
    return processed


def infer_mesh(
    processed,
    use_fixed_camera,
    camera_angle_x,
    mesh_scale,
    dense_steps,
    dense_guidance_scale,
    dense_seed,
    sparse_512_steps,
    sparse_512_guidance_scale,
    sparse_1024_steps,
    sparse_1024_guidance_scale,
    sparse_seed,
    dense_threshold,
    mc_threshold,
):
    """Step 2: run 3D inference on the already-processed image."""
    if processed is None or pipeline is None:
        return None, None

    tmp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(processed).save(tmp_input.name)
    input_path = tmp_input.name

    try:
        if use_fixed_camera:
            mesh = Pixal3DPipeline.infer(
                pipeline,
                image=input_path,
                camera_angle_x=camera_angle_x,
                mesh_scale=mesh_scale,
                dense_steps=int(dense_steps),
                dense_guidance_scale=dense_guidance_scale,
                dense_seed=int(dense_seed),
                sparse_512_steps=int(sparse_512_steps),
                sparse_512_guidance_scale=sparse_512_guidance_scale,
                sparse_1024_steps=int(sparse_1024_steps),
                sparse_1024_guidance_scale=sparse_1024_guidance_scale,
                sparse_seed=int(sparse_seed),
                dense_threshold=dense_threshold,
                mc_threshold=mc_threshold,
            )
        else:
            mesh = pipeline.infer(
                image=input_path,
                mesh_scale=mesh_scale,
                optimize_mesh_scale=True,
                target_padding=3,
                max_optim_iterations=2,
                dense_steps=int(dense_steps),
                dense_guidance_scale=dense_guidance_scale,
                dense_seed=int(dense_seed),
                sparse_512_steps=int(sparse_512_steps),
                sparse_512_guidance_scale=sparse_512_guidance_scale,
                sparse_1024_steps=int(sparse_1024_steps),
                sparse_1024_guidance_scale=sparse_1024_guidance_scale,
                sparse_seed=int(sparse_seed),
                dense_threshold=dense_threshold,
                mc_threshold=mc_threshold,
            )

        ply_file = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        ply_path = ply_file.name
        glb_path = glb_file.name
        ply_file.close()
        glb_file.close()
        mesh.export(ply_path)
        mesh.export(glb_path, file_type="glb")

        return glb_path, ply_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        os.unlink(input_path)


def build_ui():
    with gr.Blocks(title="Pixal3D", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Pixal3D: Pixel-Aligned 3D Generation from Images")

        with gr.Row():
            # Left column: input
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input Image", type="numpy", image_mode=None)

                use_rmbg = gr.Checkbox(
                    label="Remove Background",
                    value=False,
                    info="Checked: auto remove background via BiRefNet. Unchecked: use original image directly.",
                )

                use_fixed_camera = gr.Checkbox(
                    label="Use Fixed Camera Parameters",
                    value=False,
                    info="If checked, use manually set FOV/distance/mesh_scale instead of MoGe auto-estimation.",
                )

                with gr.Group(visible=False) as fixed_camera_group:
                    gr.Markdown("### Camera Parameters (fixed mode)")
                    camera_angle_x = gr.Number(value=0.2, label="camera_angle_x (rad)", step=0.01)

                with gr.Group():
                    gr.Markdown("### Mesh Scale")
                    mesh_scale = gr.Number(value=0.5, label="mesh_scale", step=0.01,
                                           info="Initial mesh scale. Fixed mode default: 0.9, Auto mode default: 0.5")

                with gr.Accordion("Advanced Inference Parameters", open=False):
                    dense_steps = gr.Number(value=50, label="Dense Steps", step=1, precision=0)
                    dense_guidance_scale = gr.Number(value=7.0, label="Dense Guidance Scale", step=0.1)
                    dense_seed = gr.Number(value=0, label="Dense Seed", step=1, precision=0)
                    sparse_512_steps = gr.Number(value=30, label="Sparse 512 Steps", step=1, precision=0)
                    sparse_512_guidance_scale = gr.Number(value=7.0, label="Sparse 512 Guidance Scale", step=0.1)
                    sparse_1024_steps = gr.Number(value=15, label="Sparse 1024 Steps", step=1, precision=0)
                    sparse_1024_guidance_scale = gr.Number(value=7.0, label="Sparse 1024 Guidance Scale", step=0.1)
                    sparse_seed = gr.Number(value=0, label="Sparse Seed", step=1, precision=0)
                    dense_threshold = gr.Number(value=0.1, label="Dense Threshold", step=0.01)
                    mc_threshold = gr.Number(value=0.2, label="MC Threshold", step=0.01)

                run_btn = gr.Button("Generate 3D Mesh", variant="primary", size="lg")

            # Right column: output
            with gr.Column(scale=1):
                processed_image = gr.Image(
                    label="Processed Image",
                    image_mode="RGBA",
                    type="numpy",
                    interactive=False,
                )
                model_viewer = gr.Model3D(label="3D Mesh Preview", camera_position=(90, 180, None))
                output_file = gr.File(label="Download .ply")

        # Toggle fixed camera group visibility and mesh_scale default
        def on_toggle_fixed(use_fixed):
            new_scale = 0.9 if use_fixed else 0.5
            return gr.update(visible=use_fixed), gr.update(value=new_scale)

        use_fixed_camera.change(
            fn=on_toggle_fixed,
            inputs=[use_fixed_camera],
            outputs=[fixed_camera_group, mesh_scale],
        )

        # Step 1: preprocess image → show processed image immediately
        # Step 2: run 3D inference → show mesh and download
        run_btn.click(
            fn=preprocess_image,
            inputs=[image_input, use_rmbg],
            outputs=[processed_image],
        ).then(
            fn=infer_mesh,
            inputs=[
                processed_image,
                use_fixed_camera,
                camera_angle_x,
                mesh_scale,
                dense_steps,
                dense_guidance_scale,
                dense_seed,
                sparse_512_steps,
                sparse_512_guidance_scale,
                sparse_1024_steps,
                sparse_1024_guidance_scale,
                sparse_seed,
                dense_threshold,
                mc_threshold,
            ],
            outputs=[model_viewer, output_file],
        )

    demo.queue(api_open=False)
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--repo_id", type=str, default="TencentARC/Pixal3D-D")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--root_path", type=str, default="",
                        help="Root path for reverse proxy ")
    args = parser.parse_args()

    load_pipeline(ckpt_dir=args.ckpt_dir, repo_id=args.repo_id)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        max_file_size="100mb",
    )
