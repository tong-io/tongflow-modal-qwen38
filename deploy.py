"""Modal deploy entry for qwen38 (Qwen3.8-27B).

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import os
import modal
from tongflow import deploy
from pathlib import Path
from typing import Any, List, Optional

_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "Qwen/Qwen3.8-27B")
MODEL_DIR = f"/models/{REPO_ID}"
_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.image_describe import ImageDescribeInput, ImageDescribeOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.video_describe import VideoDescribeInput, VideoDescribeOutput
from tongflow.models.video_gen_text import VideoGenTextInput, VideoGenTextOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import prompt_media_to_bytes
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v

# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "gen-text": {
        "thinking": {"type": "boolean", "default": False, "label": "Thinking"},
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
    "combine-text": {
        "thinking": {"type": "boolean", "default": False, "label": "Thinking"},
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
    "image-describe": {
        "thinking": {"type": "boolean", "default": False, "label": "Thinking"},
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
    "video-describe": {
        "thinking": {"type": "boolean", "default": False, "label": "Thinking"},
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
    "image-gen-text": {
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
    "video-gen-text": {
        "reasoning_effort": {"type": "select", "options": ["low", "medium", "high"], "default": "low", "label": "Reasoning effort", "description": "Applies when thinking is on."},
    },
}

# ── plugin-internal knobs (not ABI fields) ───────────────────────────────────

# Reasoning depth used whenever thinking is enabled. Qwen3.8 defaults to
# "xhigh", which is far too slow for an interactive canvas node.
REASONING_EFFORT = os.environ.get("QWEN38_REASONING_EFFORT", "low")
DEFAULT_MAX_NEW_TOKENS = 1024
# Thinking-mode answers need headroom for the <think> block itself.
THINKING_MAX_NEW_TOKENS = 8192
# Video budget: frames are resampled so the whole clip fits in this many
# frames at <= 480p before being handed to the native video processor.
VIDEO_MAX_FRAMES = 120
VIDEO_MAX_HEIGHT = 480
DESCRIBE_IMAGE_FALLBACK = "Describe this image in detail."
DESCRIBE_VIDEO_FALLBACK = "Describe this video in detail."

# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .apt_install("ffmpeg")
    .pip_install(
        "tongflow==1.0.0", "fastapi[standard]",
        "transformers==5.15.0",
        "accelerate==1.13.0",
        "torchvision",
        "av>=12.0",
        "pillow>=10.0",
        "huggingface_hub>=1.5.0,<2.0",
        # Triton kernels for the linear-attention layers (pure Python, no build).
        "flash-linear-attention",
    )
)

with image.imports():
    import re
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="H100",
    memory=16384,
    volumes={"/models": volume},
    timeout=1200,
)
class Inference:
    @modal.enter()
    def load(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )

    # ── core ─────────────────────────────────────────────────────────────

    def _infer(
        self,
        messages,
        max_new_tokens: int,
        enable_thinking: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> dict:
        template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if enable_thinking:
            template_kwargs["reasoning_effort"] = _adv("reasoning_effort", REASONING_EFFORT)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Official recommendations: thinking 1.0/0.95/20, instruct 0.7/0.8/20.
        if temperature is None:
            temperature = 1.0 if enable_thinking else 0.7
        if top_p is None:
            top_p = 0.95 if enable_thinking else 0.8
        if top_k is None:
            top_k = 20

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        thinking = ""
        m = re.search(r"<think>(.*?)</think>", response, flags=re.S)
        if m:
            thinking = m.group(1).strip()
            response = response[m.end():]
        elif "</think>" in response:
            # The opening tag is sometimes emitted by the template, not the model.
            head, _, response = response.partition("</think>")
            thinking = head.strip()
        return {"text": response.strip(), "thinking": thinking}

    def _prepare_video(self, video_bytes: bytes, tmp_files: list) -> str:
        """Resample the clip to <=VIDEO_MAX_FRAMES frames at <=480p; return path."""
        import json, subprocess, tempfile

        src = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        src.write(video_bytes)
        src.close()
        tmp_files.append(src.name)

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", src.name],
            capture_output=True, text=True,
        )
        streams = json.loads(probe.stdout or "{}").get("streams", [])
        vs = next((s for s in streams if s.get("codec_type") == "video"), None)
        duration = float(vs.get("duration", 0) or 0) if vs else 0.0
        height = int(vs.get("height", 0) or 0) if vs else 0

        fps = 1.0
        if duration > VIDEO_MAX_FRAMES:
            fps = VIDEO_MAX_FRAMES / duration
        vf = f"fps={fps:.4f}"
        if height > VIDEO_MAX_HEIGHT:
            vf += f",scale=-2:{VIDEO_MAX_HEIGHT}"

        dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        dst.close()
        tmp_files.append(dst.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", src.name, "-an", "-vf", vf,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", dst.name],
            capture_output=True, check=True,
        )
        return dst.name

    def _generate_multimodal(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        video: Optional[bytes] = None,
        system: str = "",
        max_new_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Multimodal generation from raw bytes. Returns {"text", "thinking"}."""
        import os as _os, shutil, tempfile

        if max_new_tokens is None:
            max_new_tokens = THINKING_MAX_NEW_TOKENS if enable_thinking else DEFAULT_MAX_NEW_TOKENS

        tmp_files: list[str] = []
        try:
            content: list[dict[str, Any]] = []
            for img_bytes in images or []:
                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                f.write(img_bytes)
                f.close()
                tmp_files.append(f.name)
                content.append({"type": "image", "url": f.name})
            if video:
                content.append({"type": "video", "url": self._prepare_video(video, tmp_files)})
            content.append({"type": "text", "text": prompt})

            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
            messages.append({"role": "user", "content": content})
            return self._infer(messages, max_new_tokens, enable_thinking, temperature, top_p, top_k)
        finally:
            for path in tmp_files:
                if _os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif _os.path.exists(path):
                    _os.unlink(path)

    # ── slots ────────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.GEN_TEXT)
    def gen_text(self, input: GenTextInput) -> GenTextOutput:
        user_message = (
            f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
            "Note: output only the requested answer. Do not include any other content."
        )
        out = self._generate_multimodal(prompt=user_message, enable_thinking=_adv("thinking", False))
        return GenTextOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.COMBINE_TEXT)
    def combine_text(self, input: CombineTextInput) -> CombineTextOutput:
        joined = "\n\n".join(input.texts)
        user_message = (
            f"{input.userPrompt or ''}\n\nUser input: {joined}\n\n"
            "Note: output only the requested answer. Do not include any other content."
        )
        out = self._generate_multimodal(prompt=user_message, enable_thinking=_adv("thinking", False))
        return CombineTextOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_DESCRIBE)
    def image_describe(self, input: ImageDescribeInput) -> ImageDescribeOutput:
        instruction = (
            (input.userPrompt or "").strip()
            or (input.text or "").strip()
            or DESCRIBE_IMAGE_FALLBACK
        )
        out = self._generate_multimodal(
            prompt=instruction,
            images=[prompt_media_to_bytes(input.image)],
            enable_thinking=_adv("thinking", False),
        )
        return ImageDescribeOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.VIDEO_DESCRIBE)
    def video_describe(self, input: VideoDescribeInput) -> VideoDescribeOutput:
        instruction = (
            (input.userPrompt or "").strip()
            or (input.text or "").strip()
            or DESCRIBE_VIDEO_FALLBACK
        )
        out = self._generate_multimodal(
            prompt=instruction,
            video=prompt_media_to_bytes(input.video),
            enable_thinking=_adv("thinking", False),
        )
        return VideoDescribeOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_TEXT)
    def image_gen_text(self, input: ImageGenTextInput) -> ImageGenTextOutput:
        images: Optional[List[bytes]] = None
        if input.image is not None:
            images = [prompt_media_to_bytes(input.image)]
        out = self._generate_multimodal(
            prompt=input.text,
            images=images,
            system=input.system or "",
            max_new_tokens=int(input.max_new_tokens) if input.max_new_tokens is not None else None,
            enable_thinking=bool(input.enable_thinking),
            temperature=input.temperature,
            top_p=input.top_p,
            top_k=input.top_k,
        )
        return ImageGenTextOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.VIDEO_GEN_TEXT)
    def video_gen_text(self, input: VideoGenTextInput) -> VideoGenTextOutput:
        out = self._generate_multimodal(
            prompt=input.text,
            video=prompt_media_to_bytes(input.video),
            system=input.system or "",
            max_new_tokens=int(input.max_new_tokens) if input.max_new_tokens is not None else None,
            enable_thinking=bool(input.enable_thinking),
            temperature=input.temperature,
            top_p=input.top_p,
            top_k=input.top_k,
        )
        return VideoGenTextOutput(success=True, text=str(out.get("text", "")))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
