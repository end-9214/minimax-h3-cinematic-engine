# MiniMax-H3 Cinematic Engine 🚀

Production-grade serverless video generation & multi-shot continuity API for **MiniMax-H3**, running headless **ComfyUI** on NVIDIA A100 / H100 GPU via **Modal**.

Completely project-agnostic, with zero hardcoded credentials, characters, or scene names.

---

## 🎭 Supported Workflows

| `workflow_type` | ComfyUI Source | Purpose |
|-----------------|----------------|---------|
| `base_generation` | `01_base_generation.json` | **Shot 01: Master Wide Spatial Anchor**. Establishes the scene, lighting, and wardrobe anchors. |
| `continuation` | `02_continuation.json` | **Shot 02+: Dual-Seed Latent Continuation**. Chains previous latent + tail frames while anchoring to Shot 01. |
| `closeup_masterref` | `03_closeup_masterref.json` | **Shot N: Closeup + Dual Master Ref Anchor**. Seamless medium/close-up framing anchored to master reference. |

---

## ⚡ Architecture & Latent Chaining

```
Shot 01 (base_generation)
  └─► Output: Video (MP4) + Latent (safetensors)

Shot 02 (continuation)
  ├─► prev_latent_url = Shot 01 Latent
  ├─► prev_video_url  = Shot 01 Video Tail
  └─► ref_video_url   = Shot 01 Video Anchor (Master Spatial Anchor)
        └─► Output: Video 02 + Latent 02

Shot 03 (closeup_masterref or continuation)
  ├─► prev_latent_url = Shot 02 Latent
  ├─► prev_video_url  = Shot 02 Video Tail
  ├─► ref_video_url   = Shot 01 Video Anchor
  └─► closeup_video_url = Shot 02 Video
        └─► Output: Video 03 + Latent 03
```

---

## 🛠️ Deployment Instructions

### 1. Modal Setup & Authentication
```bash
pip install modal
modal setup
```

### 2. Configure Modal Secrets (No hardcoded credentials)
```bash
modal secret create minimax-h3-secrets \
  MODAL_APP_NAME="minimax-h3-cinematic-engine" \
  MODELS_VOLUME_NAME="magicroll-h3-models" \
  OUTPUTS_VOLUME_NAME="minimax-h3-engine-outputs"
```

### 3. Deploy Serverless App
```bash
cd minimax-h3-cinematic-engine
modal deploy modal_app.py
```

After deployment, Modal outputs your REST endpoints:
- `POST /generate_shot` — Generation endpoint
- `GET /get_artifact?filename=...` — High-speed CDN artifact downloader & streaming
- `GET /ui` — Interactive ComfyUI browser interface

---

## 📡 API Specification

### `POST /generate_shot`

#### Parameters

| Field | Type | Description |
|-------|------|-------------|
| `workflow_type` | `str` | `"base_generation"`, `"continuation"`, or `"closeup_masterref"` |
| `prompt_text` | `str` | Calibrated cinematic prompt (9:16 vertical, Alexa 35 aesthetic) |
| `seed` | `int` | Generation seed |
| `image_slot_1_b64` / `image_slot_1_url` | `str` | Character sheet 1 (base64 or URL) |
| `image_slot_2_b64` / `image_slot_2_url` | `str` | Character sheet 2 (base64 or URL) |
| `image_slot_3_b64` / `image_slot_3_url` | `str` | Clean plate / environment sheet (base64 or URL) |
| `prev_latent_b64` / `prev_latent_url` | `str` | Previous shot latent safetensors (for continuation) |
| `prev_video_b64` / `prev_video_url` | `str` | Previous shot video MP4 (for continuation) |
| `ref_video_b64` / `ref_video_url` | `str` | Shot 01 master anchor video MP4 |
| `closeup_video_b64` / `closeup_video_url` | `str` | Closeup reference video (for closeup workflow) |
| `return_base64` | `bool` | Default `False`. Set `True` to include base64 bytes in JSON. |

#### Response

```json
{
  "status": "success",
  "session_id": "a93b4e12",
  "workflow_type": "continuation",
  "video_filename": "shot_a93b4e12_video_00001_.mp4",
  "latent_filename": "shot_a93b4e12_latent_00001_.safetensors",
  "video_url": "https://<app-domain>-get-artifact.us-east.modal.run?filename=shot_a93b4e12_video_00001_.mp4",
  "latent_url": "https://<app-domain>-get-artifact.us-east.modal.run?filename=shot_a93b4e12_latent_00001_.safetensors"
}
```

---

## 📂 Repository Structure

```
minimax-h3-cinematic-engine/
├── modal_app.py            # Modal serverless ComfyUI runner & REST API
├── client_example.py       # Python client with CDN chaining examples
├── scripts/
│   └── patch_torch.py      # PyTorch 2.5 PEP 585 infer_schema patch
├── workflows/
│   ├── 01_base_generation.json      # Shot 01 ComfyUI workflow
│   ├── 02_continuation.json         # Shot 02+ Continuation workflow
│   └── 03_closeup_masterref.json    # Closeup + Master Ref workflow
├── input_assets/           # Local folder for reference sheets (gitignored)
├── requirements-client.txt # Client dependencies (httpx)
├── .gitignore              # Protects models, outputs, tokens & credentials
└── README.md
```

---

## 🔒 Security & Best Practices
- Models and weights are stored in Modal persistent volumes (`magicroll-h3-models`).
- Generated artifacts and latents are isolated in `minimax-h3-engine-outputs`.
- Secrets, tokens, and credentials are never stored in the repository.
