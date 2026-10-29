#!/usr/bin/env python3
"""
🎬 Example client for the MiniMax-H3 Cinematic Engine API on Modal.

Demonstrates:
  - Shot 01: Base Generation (Shot 1 master wide spatial anchor)
  - Shot 02+: Dual-Seed Latent Continuation using CDN URLs
  - Shot 03+: Closeup + Master Ref Workflow using CDN URLs

All input images and video/latent assets can be passed as:
  - Base64 string: embed directly in request (ideal for character sheets)
  - CDN URL: direct output URL from a previous shot (ideal for video/latent chaining)
"""

import os
import sys
import json
import base64
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Load configuration from .env file if present
load_dotenv()

# Configure your deployed Modal endpoint URL (set via .env or environment variable):
ENDPOINT = os.environ.get(
    "MINIMAX_H3_ENDPOINT",
    "https://YOUR_MODAL_WORKSPACE--minimax-h3-cinematic-engine-generate-shot.us-east.modal.run"
)

def encode_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_generate(payload: dict) -> dict:
    print(f"\n🚀 Invoking MiniMax-H3 Engine | Workflow: {payload.get('workflow_type')} | Seed: {payload.get('seed')}")
    with httpx.Client(timeout=1800) as client:
        resp = client.post(ENDPOINT, json=payload)
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Shot 01: Base Generation (Master Anchor)
# ─────────────────────────────────────────────────────────────────────────────
def generate_shot_01(
    prompt_text: str,
    char1_sheet_path: str,
    char2_sheet_path: str,
    clean_plate_path: str,
    seed: int = 1007,
) -> dict:
    payload = {
        "workflow_type": "base_generation",
        "prompt_text": prompt_text,
        "seed": seed,
        "return_base64": False,  # CDN URLs returned by default
        "image_slot_1_b64": encode_file_b64(char1_sheet_path),
        "image_slot_1_name": Path(char1_sheet_path).name,
        "image_slot_2_b64": encode_file_b64(char2_sheet_path),
        "image_slot_2_name": Path(char2_sheet_path).name,
        "image_slot_3_b64": encode_file_b64(clean_plate_path),
        "image_slot_3_name": Path(clean_plate_path).name,
    }
    res = call_generate(payload)
    print("✅ Shot 01 Completed!")
    print(f"   Video URL  : {res.get('video_url')}")
    print(f"   Latent URL : {res.get('latent_url')}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shot 02+: Continuation via CDN URLs
# ─────────────────────────────────────────────────────────────────────────────
def generate_continuation_shot(
    prompt_text: str,
    char1_sheet_path: str,
    char2_sheet_path: str,
    clean_plate_path: str,
    prev_shot_result: dict,
    shot1_anchor_result: dict,
    seed: int = 1014,
) -> dict:
    payload = {
        "workflow_type": "continuation",
        "prompt_text": prompt_text,
        "seed": seed,
        "return_base64": False,
        "image_slot_1_b64": encode_file_b64(char1_sheet_path),
        "image_slot_1_name": Path(char1_sheet_path).name,
        "image_slot_2_b64": encode_file_b64(char2_sheet_path),
        "image_slot_2_name": Path(char2_sheet_path).name,
        "image_slot_3_b64": encode_file_b64(clean_plate_path),
        "image_slot_3_name": Path(clean_plate_path).name,
        # Chained continuation assets via CDN URLs
        "prev_latent_url": prev_shot_result["latent_url"],
        "prev_video_url": prev_shot_result["video_url"],
        "ref_video_url": shot1_anchor_result["video_url"],  # Shot 1 master wide anchor
    }
    res = call_generate(payload)
    print("✅ Continuation Shot Completed!")
    print(f"   Video URL  : {res.get('video_url')}")
    print(f"   Latent URL : {res.get('latent_url')}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shot N: Closeup + Master Ref Workflow
# ─────────────────────────────────────────────────────────────────────────────
def generate_closeup_shot(
    prompt_text: str,
    char1_sheet_path: str,
    char2_sheet_path: str,
    clean_plate_path: str,
    prev_shot_result: dict,
    shot1_anchor_result: dict,
    seed: int = 1021,
) -> dict:
    payload = {
        "workflow_type": "closeup_masterref",
        "prompt_text": prompt_text,
        "seed": seed,
        "return_base64": False,
        "image_slot_1_b64": encode_file_b64(char1_sheet_path),
        "image_slot_1_name": Path(char1_sheet_path).name,
        "image_slot_2_b64": encode_file_b64(char2_sheet_path),
        "image_slot_2_name": Path(char2_sheet_path).name,
        "image_slot_3_b64": encode_file_b64(clean_plate_path),
        "image_slot_3_name": Path(clean_plate_path).name,
        # Chaining
        "prev_latent_url": prev_shot_result["latent_url"],
        "prev_video_url": prev_shot_result["video_url"],
        "ref_video_url": shot1_anchor_result["video_url"],
        # Closeup reference
        "closeup_video_url": prev_shot_result["video_url"],
    }
    res = call_generate(payload)
    print("✅ Closeup Shot Completed!")
    print(f"   Video URL  : {res.get('video_url')}")
    print(f"   Latent URL : {res.get('latent_url')}")
    return res


if __name__ == "__main__":
    print("Example client initialized. Import and invoke functions with your scene assets.")
