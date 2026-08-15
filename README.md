# tongflow-modal-qwen38

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Multimodal text with **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`, Apache-2.0), running on an H100 via [Modal](https://modal.com). Generates / rewrites text, and describes or answers questions about images and video.

## Capabilities

- **Text generation** (`gen-text`) — generate or rewrite text from a prompt.
- **Combine text** (`combine-text`) — merge several text inputs under one instruction.
- **Image understanding** (`image-describe`, `image-gen-text`) — captions, Q&A, or descriptions from an image.
- **Video understanding** (`video-describe`, `video-gen-text`) — summaries or descriptions from a video (native video input, resampled to ≤120 frames at ≤480p).

Thinking mode is **off** by default (instruct sampling `0.7 / 0.8 / 20`). The `image-gen-text` / `video-gen-text` nodes expose an `enable_thinking` toggle; when on, the plugin uses `reasoning_effort` `low` (override with the `QWEN38_REASONING_EFFORT` env var: `low` / `medium` / `xhigh`) and thinking-mode sampling `1.0 / 0.95 / 20`.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The `Qwen/Qwen3.8-27B` weights (~56 GB, BF16) are public — no Hugging Face token required; fetch them into the `models` volume once with `modal run download.py::download`.
