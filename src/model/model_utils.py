"""
Utilities for loading models, processors, and preparing visual inputs for Qwen2.5-VL.
"""

from __future__ import annotations

import torch


# ─── Model-type detection ─────────────────────────────────────────────────────
def get_model_type(model_id: str) -> str:
    """Detect the backbone type from a model path or HuggingFace model ID."""
    name = model_id.lower()
    if "qwen2.5" in name or "qwen2_5" in name or "VideoChat-R1_5" in model_id:
        return "qwen2_5_vl"
    raise ValueError(
        f"Cannot auto-detect backbone type for model_id='{model_id}'. "
        "Supported patterns: 'Qwen2.5' / 'Qwen2_5'. "
        "Pass --model_type explicitly if your path does not match."
    )


# ─── Model factory ────────────────────────────────────────────────────────────

def load_model(model_id: str, model_type: str, **kwargs):
    """Load Qwen2.5-VL from a checkpoint."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    kwargs.pop("trust_remote_code", None)
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)


# ─── Processor factory ────────────────────────────────────────────────────────

def load_processor(
    model_id: str,
    model_type: str,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
):
    """Load and configure the Qwen2.5-VL processor."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)

    # Normalize special token IDs so training code can access them uniformly.
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    processor.pad_token_id = pad_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    if hasattr(processor, "image_processor"):
        if max_pixels is not None:
            processor.image_processor.max_pixels = max_pixels
        if min_pixels is not None:
            processor.image_processor.min_pixels = min_pixels

    return processor


# ─── Conversation / message helpers ──────────────────────────────────────────

def make_video_content_element(
    model_type: str,
    video_path: str,
    video_start: float | None = None,
    video_end: float | None = None,
    total_pixels: int = 3584 * 28 * 28,
    min_pixels: int = 16 * 28 * 28,
) -> dict:
    """Build the content-element dict for a video inside a chat message."""
    ele: dict = {
        "type": "video",
        "video": video_path,
        "total_pixels": total_pixels,
        "min_pixels": min_pixels,
    }
    if video_start is not None:
        ele["video_start"] = video_start
    if video_end is not None:
        ele["video_end"] = video_end
    return ele


# ─── Processor call abstraction ───────────────────────────────────────────────

def call_processor_for_video(
    processor,
    model_type: str,
    texts: list[str],
    video_inputs,
    fps_inputs,
    video_metadatas=None,
    **kwargs,
):
    """Call the processor with backbone-appropriate keyword arguments."""
    return processor(
        text=texts,
        images=None,
        videos=video_inputs,
        fps=fps_inputs,
        **kwargs,
    )


# ─── Vision kwargs extraction / repetition ────────────────────────────────────

def extract_vision_kwargs(model_type: str, prompt_inputs) -> dict:
    """Extract vision-specific tensors from processor output."""
    out: dict = {}
    if "pixel_values_videos" in prompt_inputs:
        out["pixel_values_videos"] = prompt_inputs["pixel_values_videos"]
    if "video_grid_thw" in prompt_inputs:
        out["video_grid_thw"] = prompt_inputs["video_grid_thw"]
    return out


def repeat_vision_kwargs(
    model_type: str,
    vision_kwargs: dict,
    num_generations: int,
) -> dict:
    """Tile vision tensors to match num_generations repeated completions.

    For Qwen2.5-VL, `pixel_values_videos` is a flat concatenation of all B
    videos' patch tokens.  We use `video_grid_thw` (shape: B×3) to compute
    each video's patch count, split the tensor, repeat each chunk G times,
    and re-concatenate in the correct order.
    """
    if num_generations == 1:
        return dict(vision_kwargs)

    G = num_generations
    out: dict = {}

    if "video_grid_thw" in vision_kwargs:
        video_grid_thw = vision_kwargs["video_grid_thw"]          # (B, 3)
        out["video_grid_thw"] = video_grid_thw.repeat_interleave(G, dim=0)

    if "pixel_values_videos" in vision_kwargs:
        pixel_values_videos = vision_kwargs["pixel_values_videos"] # (Σn_i, D)

        if "video_grid_thw" in vision_kwargs:
            video_grid_thw = vision_kwargs["video_grid_thw"]       # (B, 3)
            patch_counts = (
                video_grid_thw[:, 0]
                * video_grid_thw[:, 1]
                * video_grid_thw[:, 2]
            ).tolist()
            patch_counts = [int(n) for n in patch_counts]

            per_video = torch.split(pixel_values_videos, patch_counts, dim=0)
            out["pixel_values_videos"] = torch.cat(
                [chunk.repeat(G, 1) for chunk in per_video], dim=0
            )
        else:
            # B=1 fallback (no grid info available)
            out["pixel_values_videos"] = pixel_values_videos.repeat(G, 1)

    return out


# ─── vLLM stop-token IDs ─────────────────────────────────────────────────────

def get_vllm_stop_token_ids(model_type: str, tokenizer) -> list[int]:
    """Return stop-token IDs for Qwen2.5-VL: <|im_end|> and <|endoftext|>."""
    return [151645, 151643]
