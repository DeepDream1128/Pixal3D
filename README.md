
<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<div align="center">
    <img src="assets/teaser.png" alt="Teaser image of Pixal3D"/>
</div>

**Pixal3D** generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

---

## ✨ News

- **May 2026**: Release the improved version based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone. 💪
- **May 2026**: Release inference code and online demo. 🤗
- **Apr 2026**: Our paper is accepted to SIGGRAPH 2026! 🎉

## 📌 Branches

| Branch | Description |
|--------|-------------|
| `main` | **Latest version** — improved implementation based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone with better performance. |
| `paper` | **Paper version** — original implementation based on [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), corresponding to results reported in our SIGGRAPH 2026 paper. |

> If you want to reproduce the results in our paper, please switch to the `paper` branch.



## 🚀 Getting Started

### Installation

#### Tested Environment

- **System**: Ubuntu 22.04
- **CUDA Toolkit**: CUDA 12.4
- **Python**: 3.10

#### Step 1: Create Environment

```bash
conda create -n pixal3d python=3.10
conda activate pixal3d
conda install bioconda::google-sparsehash -y
```

#### Step 2: Install PyTorch

Make sure the PyTorch CUDA version matches your installed CUDA Toolkit.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Step 3: Install Torchsparse

Follow the [official guide](https://github.com/mit-han-lab/torchsparse) or:

```bash
git clone https://github.com/mit-han-lab/torchsparse
cd torchsparse && python -m pip install .
cd ..
```

#### Step 4: Install Pixal3D

```bash
git clone https://github.com/Pixal3D-open/Pixal3D.git
cd Pixal3D

pip install -r requirements.txt

pip install third_party/voxelize
```

### Usage



#### Basic Inference (Fixed Camera)

```python
from pixal3dpipeline import Pixal3DPipeline

pipeline = Pixal3DPipeline.from_pretrained(repo_id="TencentARC/Pixal3D-D")

mesh = pipeline.infer_from_image(
    image_path="assets/test_image/0.png",
    camera_angle_x=0.2,
    mesh_scale=0.9,
)

mesh.export("output.ply")
```

Or via command line:

```bash
python inference.py \
    --image assets/test_image/0.png \
    --output ./outputs
```

#### 2-Stage Inference (with MoGe FOV Estimation)

The 2-stage pipeline uses [MoGe](https://github.com/microsoft/MoGe) to automatically estimate camera FOV from the input image, and iteratively optimizes `mesh_scale` for better results.

```python
from pixal3dpipeline2stage import Pixal3DPipeline2Stage

pipeline = Pixal3DPipeline2Stage.from_pretrained(
    ckpt_dir="./ckpt",
    repo_id="TencentARC/Pixal3D-D",
    use_moge=True,
    use_dense_check=True,
)

mesh = pipeline.infer_from_image(
    image_path="assets/test_image/0.png",
    mesh_scale=0.5,
    optimize_mesh_scale=True,
    target_padding=3,
    max_optim_iterations=2,
)

mesh.export("output.ply")
```

Or via command line:

```bash
python inference2stage.py \
    --image assets/test_image/0.png \
    --output ./outputs_2stage \
    --ckpt_dir ./ckpt \
    --mesh_scale 0.5 \
    --target_padding 3 \
    --max_optim_iterations 2 \
    --dense_steps 50 \
    --dense_guidance_scale 7.0 \
    --sparse_512_steps 30 \
    --sparse_512_guidance_scale 7.0 \
    --sparse_1024_steps 15 \
    --sparse_1024_guidance_scale 7.0
```

### Web Demo

We provide a Gradio web demo for Pixal3D, which allows you to generate 3D meshes from images interactively.

```bash
python gradio_app.py  --port 12345
```

## 🤗 Acknowledgements

This project is heavily built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2) and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We sincerely thank the authors for their outstanding work on scalable 3D generation , which serves as the foundation of our codebase and model architecture.

We also thank the following repos for their great contributions:

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
