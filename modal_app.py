"""
🎬 MiniMax-H3 Cinematic Engine — Production Modal API
A clean, project-agnostic, GitHub-ready serverless video generation & multi-shot continuity engine.

Workflows supported:
  - base_generation    (Shot 01: Master Wide Spatial Anchor)
  - continuation       (Shot 02+: Dual-Seed Latent Continuation)
  - closeup_masterref  (Shot N: Closeup + Dual Master Ref Anchor)

Features:
  - Zero hardcoded names, paths, or credentials.
  - Accepts assets via base64 OR CDN URLs (no re-uploading large latents/videos).
  - Returns generated video and new latent (URL + base64) for seamless shot chaining.
  - ComfyUI headless runner on NVIDIA A100-80GB / H100 GPU.
  - Interactive Web UI endpoint for visual workflow inspection.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import urllib.request
import base64
import uuid
from typing import Optional, Dict, Any
import modal

# ---------------------------------------------------------------------------
# 1. Modal App Definition & Persistent Volumes
# ---------------------------------------------------------------------------
APP_NAME = os.environ.get("MODAL_APP_NAME", "minimax-h3-cinematic-engine")
app = modal.App(APP_NAME)

MODELS_VOLUME_NAME = os.environ.get("MODELS_VOLUME_NAME", "magicroll-h3-models")
OUTPUTS_VOLUME_NAME = os.environ.get("OUTPUTS_VOLUME_NAME", "minimax-h3-engine-outputs")

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)

GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "A100-80GB")

# ---------------------------------------------------------------------------
# 2. Container Image Build
# ---------------------------------------------------------------------------
studio_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "curl", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "spandrel",
        "fastapi[standard]",
        "pydantic>=2.0",
        "requests",
        "safetensors",
        "aiohttp",
        "httpx",
    )
    .run_commands(
        # Clone ComfyUI base
        "git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "pip install -r /root/ComfyUI/requirements.txt",
        # Clone required custom nodes
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone https://github.com/rgthree/rgthree-comfy.git /root/ComfyUI/custom_nodes/rgthree-comfy",
        "git clone https://github.com/city96/ComfyUI-GGUF.git /root/ComfyUI/custom_nodes/ComfyUI-GGUF",
        "git clone https://github.com/SatoDive/Minimax-H3-Latent-Continuation.git /root/ComfyUI/custom_nodes/Minimax-H3-Latent-Continuation",
        "git clone https://github.com/M1kep/ComfyLiterals.git /root/ComfyUI/custom_nodes/ComfyLiterals",
        "git clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo.git /root/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo",
        # Setup directories
        "mkdir -p /root/models/diffusion_models /root/models/text_encoders /root/models/vae /root/models/loras /root/output/h3_latents /root/ComfyUI/input /root/workflows",
    )
    .add_local_file("./scripts/patch_torch.py", remote_path="/root/patch_torch.py")
    .run_commands(
        "python3 /root/patch_torch.py",
        "PYTHONPATH=/root/ComfyUI python3 -c 'import comfy.utils; print(\"ComfyUI core imports successfully!\")'",
    )
    .add_local_dir("./workflows", remote_path="/root/workflows")
)

# ---------------------------------------------------------------------------
# 3. Model path configuration
# ---------------------------------------------------------------------------
_EXTRA_YAML = """comfyui:
    base_path: /root/models
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    vae: vae
    loras: loras
    upscale_models: upscale_models
"""

# ---------------------------------------------------------------------------
# 4. Headless ComfyUI Server Runner
# ---------------------------------------------------------------------------
class ComfyServer:
    COMFY_URL = "http://127.0.0.1:8188"

    _SYMLINKS = [
        ("/root/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
         "/root/models/text_encoders/Minimax_H3_Qwen3vl_32b__nvfp4_awq.safetensors"),
        ("/root/models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
         "/root/models/loras/Minimax_H3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
        ("/root/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "/root/models/diffusion_models/Minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        ("/root/models/vae/minimax_h3_audio_vae_fp32.safetensors",
         "/root/models/vae/Minimax_h3_audio_vae_fp32.safetensors"),
        ("/root/models/vae/minimax_h3_video_vae_int8_convrot.safetensors",
         "/root/models/vae/Minimax_h3_video_vae_int8_convrot.safetensors"),
    ]

    def __init__(self):
        self._proc = None

    def start(self):
        with open("/root/extra_models.yaml", "w") as f:
            f.write(_EXTRA_YAML)

        for src, dst in self._SYMLINKS:
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass

        try:
            urllib.request.urlopen(f"{self.COMFY_URL}/system_stats", timeout=1)
            print("⚡ ComfyUI backend already active.")
            return
        except Exception:
            pass

        cmd = [
            sys.executable, "/root/ComfyUI/main.py",
            "--listen", "127.0.0.1", "--port", "8188",
            "--extra-model-paths-config", "/root/extra_models.yaml",
            "--output-directory", "/root/output",
        ]
        self._proc = subprocess.Popen(cmd)
        for _ in range(90):
            try:
                urllib.request.urlopen(f"{self.COMFY_URL}/system_stats", timeout=2)
                print("✅ ComfyUI backend server started successfully.")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError("ComfyUI server failed to start within 90 seconds.")


def _comfy_queue(prompt_dict: dict) -> str:
    data = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI prompt queuing failed [{e.code}]: {e.read().decode()}")


def _comfy_wait(prompt_id: str, timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    print(f"⏳ Monitoring generation for prompt {prompt_id}...")
    while time.time() < deadline:
        time.sleep(8)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as r:
                hist = json.loads(r.read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {}).get("status_str", "")
                if status == "success" or "outputs" in hist[prompt_id]:
                    print("✅ Generation completed successfully.")
                    return
                if status == "error":
                    raise RuntimeError(f"ComfyUI error: {hist[prompt_id]['status'].get('messages')}")
        except RuntimeError:
            raise
        except Exception:
            pass
    raise TimeoutError(f"Generation timed out after {timeout}s.")


# ---------------------------------------------------------------------------
# 5. Core Workflow Execution Engine
# ---------------------------------------------------------------------------
WORKFLOW_MAP = {
    "base_generation":   "/root/workflows/01_base_generation.json",
    "continuation":      "/root/workflows/02_continuation.json",
    "closeup_masterref": "/root/workflows/03_closeup_masterref.json",
}

_EXCLUDE_NODES = {"18", "250", "197", "145", "264", "336", "374", "335", "375", "309"}


def _write_asset(data_bytes: bytes, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data_bytes)


def _fetch_url(url: str) -> bytes:
    import httpx
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def _resolve_asset(
    b64_data: Optional[str],
    cdn_url: Optional[str],
    dest_path: str,
) -> Optional[str]:
    if b64_data:
        _write_asset(base64.b64decode(b64_data), dest_path)
        return os.path.basename(dest_path)
    if cdn_url:
        _write_asset(_fetch_url(cdn_url), dest_path)
        return os.path.basename(dest_path)
    return None


def execute_workflow(
    *,
    workflow_type: str,
    prompt_text: str,
    seed: int,
    image_slot_1_b64: Optional[str] = None,
    image_slot_1_url: Optional[str] = None,
    image_slot_1_name: str = "slot1.jpg",
    image_slot_2_b64: Optional[str] = None,
    image_slot_2_url: Optional[str] = None,
    image_slot_2_name: str = "slot2.jpg",
    image_slot_3_b64: Optional[str] = None,
    image_slot_3_url: Optional[str] = None,
    image_slot_3_name: str = "slot3.jpg",
    prev_latent_b64: Optional[str] = None,
    prev_latent_url: Optional[str] = None,
    prev_video_b64: Optional[str] = None,
    prev_video_url: Optional[str] = None,
    ref_video_b64: Optional[str] = None,
    ref_video_url: Optional[str] = None,
    closeup_video_b64: Optional[str] = None,
    closeup_video_url: Optional[str] = None,
    return_base64: bool = False,
    session_id: Optional[str] = None,
) -> dict:
    run_id = session_id or uuid.uuid4().hex[:8]
    output_dir = "/root/output"
    latent_dir = "/root/output/h3_latents"
    input_dir = "/root/ComfyUI/input"
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    is_continuation = (workflow_type != "base_generation")

    # 1. Resolve image slots
    slot1 = _resolve_asset(image_slot_1_b64, image_slot_1_url, os.path.join(input_dir, image_slot_1_name)) or image_slot_1_name
    slot2 = _resolve_asset(image_slot_2_b64, image_slot_2_url, os.path.join(input_dir, image_slot_2_name)) or image_slot_2_name
    slot3 = _resolve_asset(image_slot_3_b64, image_slot_3_url, os.path.join(input_dir, image_slot_3_name)) or image_slot_3_name

    # 2. Resolve continuation latent & video assets
    latent_fname = None
    if is_continuation:
        latent_fname = _resolve_asset(
            prev_latent_b64, prev_latent_url,
            os.path.join(latent_dir, f"prev_latent_{run_id}.safetensors"),
        )

    prev_video_fname = None
    if is_continuation:
        prev_video_fname = _resolve_asset(
            prev_video_b64, prev_video_url,
            os.path.join(input_dir, f"prev_video_{run_id}.mp4"),
        )

    ref_video_fname = None
    if is_continuation:
        ref_video_fname = _resolve_asset(
            ref_video_b64, ref_video_url,
            os.path.join(input_dir, f"ref_video_{run_id}.mp4"),
        )

    closeup_video_fname = None
    if workflow_type == "closeup_masterref":
        closeup_video_fname = _resolve_asset(
            closeup_video_b64, closeup_video_url,
            os.path.join(input_dir, f"closeup_video_{run_id}.mp4"),
        )

    # 3. Load workflow template
    wf_path = WORKFLOW_MAP.get(workflow_type)
    if not wf_path or not os.path.exists(wf_path):
        raise ValueError(f"Unknown workflow: {workflow_type}. Options: {list(WORKFLOW_MAP.keys())}")

    with open(wf_path) as f:
        wf = json.load(f)

    links = {str(lnk[0]): [str(lnk[1]), lnk[2]] for lnk in wf.get("links", [])}
    exclude = set(_EXCLUDE_NODES)
    if not is_continuation or not latent_fname:
        exclude.add("356")

    prompt: dict = {}
    for node in wf.get("nodes", []):
        nid = str(node["id"])
        ntype = node.get("type", "")
        if nid in exclude or ntype == "Note":
            continue

        inputs: dict = dict(node.get("widgets_values_named", {}))
        for in_def in node.get("inputs", []):
            name = in_def["name"]
            lid = in_def.get("link")
            if lid is not None:
                lid = str(lid)
                if lid in links:
                    src = links[lid][0]
                    if src not in exclude:
                        inputs[name] = links[lid]

        # Config overrides
        if nid == "191":
            inputs["clip_name"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        elif nid == "190":
            inputs["unet_name"] = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        elif nid == "193":
            inputs["vae_name"] = "minimax_h3_audio_vae_fp32.safetensors"
        elif nid == "192":
            inputs["vae_name"] = "minimax_h3_video_vae_int8_convrot.safetensors"
        elif nid == "339":
            inputs["model"] = ["204", 0]
            inputs["lora_name"] = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        elif nid == "293":
            inputs["model"] = ["339", 0]
            inputs["scheduler"] = "simple"
            inputs["steps"] = 7
            inputs["denoise"] = 1.0
        elif nid == "343":
            inputs["model"] = ["339", 0]
        elif nid == "291":
            inputs["seed"] = seed
        elif nid == "294":
            inputs["sampler_name"] = "res_multistep"
        elif nid == "328":
            inputs["prompt"] = prompt_text
            inputs["resolution"] = "480P"
            inputs["aspect_ratio"] = "9:16"
            inputs["custom_ratio"] = "9:16"
            inputs["width"] = 480
            inputs["height"] = 848
            inputs["seconds"] = 10.0
            inputs["segment_seconds"] = "10.0"
            inputs["context_length"] = 22
            inputs["continuity_mode"] = "Native Guide"
            inputs["transition_seconds"] = 0.0
            inputs["advanced"] = False
            inputs["fps"] = 24
            inputs["keyframe_role"] = "First frame priority"
            inputs["ref_image_size"] = "match"
            inputs["reference_mention_mode"] = "By index"
            inputs["media_1"] = ["401", 0]
            inputs["media_type_1"] = "image"
            inputs["media_2"] = ["402", 0]
            inputs["media_type_2"] = "image"
            inputs["media_3"] = ["403", 0]
            inputs["media_type_3"] = "image"
            if is_continuation and latent_fname:
                inputs["seed_latent"] = ["356", 0]
            if is_continuation and prev_video_fname:
                inputs["seed_video"] = ["375", 3]
            if is_continuation and ref_video_fname:
                inputs["seed_ref_video"] = ["266", 0]
        elif nid == "356":
            if latent_fname:
                inputs["latent_file"] = latent_fname
        elif nid == "355":
            inputs["filename_prefix"] = f"shot_{run_id}_latent"
        elif nid == "373":
            inputs["filename_prefix"] = f"shot_{run_id}_video"
        elif nid == "374":
            inputs["filename_prefix"] = f"shot_{run_id}_stitched"

        prompt[nid] = {"class_type": ntype, "inputs": inputs}

    prompt["401"] = {"class_type": "LoadImage", "inputs": {"image": slot1}}
    prompt["402"] = {"class_type": "LoadImage", "inputs": {"image": slot2}}
    prompt["403"] = {"class_type": "LoadImage", "inputs": {"image": slot3}}

    if is_continuation and prev_video_fname:
        prompt["264"] = {"class_type": "LoadVideo", "inputs": {"file": prev_video_fname, "upload": "video"}}
        prompt["265"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["264", 0]}}
        prompt["309"] = {"class_type": "Float", "inputs": {"Number": "8"}}
        prompt["375"] = {
            "class_type": "MiniMaxH3EasyVideoTailSlicer_SatoDive",
            "inputs": {"video": ["265", 0], "slice_mode": "last_seconds", "seconds": ["309", 0], "fps": 24.0},
        }

    if is_continuation and ref_video_fname:
        prompt["266"] = {"class_type": "LoadVideo", "inputs": {"file": ref_video_fname, "upload": "video"}}

    if workflow_type == "closeup_masterref" and closeup_video_fname:
        prompt["382"] = {"class_type": "LoadVideo", "inputs": {"file": closeup_video_fname, "upload": "video"}}
        prompt["379"] = {
            "class_type": "MiniMaxH3EasyVideoTailSlicer_SatoDive",
            "inputs": {"video": ["382", 0], "slice_mode": "last_seconds", "seconds": 5.0, "fps": 24.0},
        }

    pid = _comfy_queue(prompt)
    print(f"🚀 Queued prompt {pid} | workflow={workflow_type} | seed={seed}")
    _comfy_wait(pid)

    outputs_volume.commit()

    v_prefix = f"shot_{run_id}_video"
    v_matches = [f for f in os.listdir(output_dir) if f.startswith(v_prefix) and f.endswith(".mp4")]
    v_name = sorted(v_matches)[-1] if v_matches else None

    l_prefix = f"shot_{run_id}_latent"
    l_matches = [f for f in os.listdir(latent_dir) if f.startswith(l_prefix) and f.endswith(".safetensors")]
    l_name = sorted(l_matches)[-1] if l_matches else None

    app_domain = os.environ.get("MODAL_APP_DOMAIN")
    def _cdn(fname):
        if not fname or not app_domain:
            return None
        return f"https://{app_domain}-get-artifact.us-east.modal.run?filename={fname}"

    res = {
        "status": "success",
        "session_id": run_id,
        "workflow_type": workflow_type,
        "video_filename": v_name,
        "latent_filename": l_name,
        "video_url": _cdn(v_name),
        "latent_url": _cdn(l_name),
    }

    if return_base64:
        if v_name:
            vp = os.path.join(output_dir, v_name)
            if os.path.exists(vp):
                with open(vp, "rb") as f:
                    res["video_base64"] = base64.b64encode(f.read()).decode()
        if l_name:
            lp = os.path.join(latent_dir, l_name)
            if os.path.exists(lp):
                with open(lp, "rb") as f:
                    res["latent_base64"] = base64.b64encode(f.read()).decode()

    return res


# ---------------------------------------------------------------------------
# 6. HTTP REST API: POST /generate_shot
# ---------------------------------------------------------------------------
@app.function(
    image=studio_image,
    gpu=GPU_TYPE,
    timeout=1800,
    volumes={
        "/root/models": models_volume,
        "/root/output": outputs_volume,
    },
)
@modal.fastapi_endpoint(method="POST")
def generate_shot(req: dict) -> dict:
    server = ComfyServer()
    server.start()
    return execute_workflow(**req)


# ---------------------------------------------------------------------------
# 7. HTTP Artifact Retrieval: GET /get_artifact?filename=...
# ---------------------------------------------------------------------------
@app.function(
    image=studio_image,
    volumes={"/root/output": outputs_volume},
)
@modal.fastapi_endpoint(method="GET")
def get_artifact(filename: str):
    from fastapi.responses import FileResponse, JSONResponse

    output_dir = "/root/output"
    latent_dir = os.path.join(output_dir, "h3_latents")

    for base in (output_dir, latent_dir):
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            media_type = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
            return FileResponse(candidate, media_type=media_type, filename=filename)

    return JSONResponse({"error": f"File '{filename}' not found in output volume."}, status_code=404)


# ---------------------------------------------------------------------------
# 8. Interactive ComfyUI Web UI Endpoint
# ---------------------------------------------------------------------------
@app.function(
    image=studio_image,
    gpu=GPU_TYPE,
    timeout=7200,
    volumes={
        "/root/models": models_volume,
        "/root/output": outputs_volume,
    },
)
@modal.web_server(port=8188, startup_timeout=120)
def ui():
    cmd = f"{sys.executable} /root/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --output-directory /root/output"
    subprocess.Popen(cmd, shell=True)
