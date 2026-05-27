"""SAM3 text-prompt 驱动的去背景实现。

提供一个与 ``BiRefNet`` 接口完全一致的 ``SAM3Rembg`` 类（``to`` /
``cuda`` / ``cpu`` / ``__call__(PIL.Image) -> PIL.Image RGBA``），可以无缝
替换 ``Pixal3DImageTo3DPipeline.rembg_model``。

模型权重默认从本地 ``/g12213021yx/BoyuanZhao/20251126SAM3/sam3.pt``
加载；prompt 默认为 ``"plant"``。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image


# SAM3 仓库源码路径（包含 ``sam3/`` 包）。允许通过环境变量覆盖。
_SAM3_DEFAULT_SOURCE = "/g12213021yx/BoyuanZhao/20251126SAM3/sam3"
_SAM3_SOURCE_PATH = os.environ.get("SAM3_PYTHON_PATH", _SAM3_DEFAULT_SOURCE)
if _SAM3_SOURCE_PATH and os.path.isdir(_SAM3_SOURCE_PATH) and _SAM3_SOURCE_PATH not in sys.path:
    sys.path.insert(0, _SAM3_SOURCE_PATH)


_SAM3_DEFAULT_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT_PATH",
    "/g12213021yx/BoyuanZhao/20251126SAM3/sam3.pt",
)


__all__ = ["SAM3Rembg"]


class SAM3Rembg:
    """基于 SAM3（Image，文本 grounding）做背景去除。

    Args:
        checkpoint_path: SAM3 image 模型权重 ``.pt`` 路径。
        prompt: 文本 prompt，默认 ``"plant"``。
        bpe_path: BPE 词表路径。``None`` 时自动从 ``sam3`` 包的
            ``assets/bpe_simple_vocab_16e6.txt.gz`` 读取。
        confidence_threshold: 实例检测置信度阈值，传给 ``Sam3Processor``。
        resolution: SAM3 内部预处理分辨率，默认 1008（与官方 image example 一致）。
        autocast_dtype: 推理时 CUDA autocast 使用的 dtype。
    """

    def __init__(
        self,
        checkpoint_path: str = _SAM3_DEFAULT_CHECKPOINT,
        prompt: str = "plant",
        bpe_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if bpe_path is None:
            import sam3 as _sam3
            bpe_path = os.path.join(
                os.path.dirname(_sam3.__file__),
                "..",
                "assets",
                "bpe_simple_vocab_16e6.txt.gz",
            )

        self.checkpoint_path = checkpoint_path
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.autocast_dtype = autocast_dtype

        # 模型先放在 CPU 上构建；外部（pipeline）会按 low_vram 策略再调用 ``to()``。
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )

        self._device = torch.device("cpu")
        self._processor_cls = Sam3Processor
        self._processor: Optional["Sam3Processor"] = None  # type: ignore[name-defined]

    def _ensure_processor(self):
        if self._processor is None:
            self._processor = self._processor_cls(
                self.model,
                resolution=self.resolution,
                device=str(self._device),
                confidence_threshold=self.confidence_threshold,
            )
        return self._processor

    def to(self, device) -> "SAM3Rembg":
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self._device = device
        # processor 内部缓存了 device 与 model 引用，换设备后让它重建。
        self._processor = None
        return self

    def cuda(self) -> "SAM3Rembg":
        return self.to("cuda")

    def cpu(self) -> "SAM3Rembg":
        return self.to("cpu")

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        processor = self._ensure_processor()

        device_type = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=(device_type == "cuda"),
        ):
            state = processor.set_image(rgb)
            state = processor.set_text_prompt(prompt=self.prompt, state=state)

        H = state["original_height"]
        W = state["original_width"]
        masks_soft = state.get("masks_logits", None)

        if masks_soft is None or masks_soft.numel() == 0 or masks_soft.shape[0] == 0:
            print(
                f"[SAM3Rembg] warning: no instances matched prompt='{self.prompt}'. "
                "Falling back to fully-opaque alpha (no background removal)."
            )
            mask_pil = Image.new("L", (W, H), color=255)
        else:
            # masks_soft: [K, 1, H, W] 已经过 sigmoid，值域 [0,1]
            soft = masks_soft.float().squeeze(1)        # [K, H, W]
            union = soft.amax(dim=0)                     # [H, W] 多实例并集（按概率取 max）
            union_np = union.detach().cpu().clamp(0, 1).numpy()
            mask_pil = Image.fromarray((union_np * 255.0).astype(np.uint8), mode="L")

        out = rgb.copy()
        out.putalpha(mask_pil)
        return out
