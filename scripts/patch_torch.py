#!/usr/bin/env python3
"""
Patch PyTorch 2.5 infer_schema to support PEP 585 list[T] type annotations in custom ops.
Required for ComfyUI custom nodes running on PyTorch 2.5+.
"""
import os
import sys

def patch_torch():
    p = "/usr/local/lib/python3.12/site-packages/torch/_library/infer_schema.py"
    if not os.path.exists(p):
        print(f"⚠️ PyTorch infer_schema not found at {p}")
        return

    with open(p, "r") as f:
        content = f.read()

    target = "if annotation_type not in SUPPORTED_PARAM_TYPES.keys():"
    replacement = (
        'if getattr(annotation_type, "__origin__", None) is list and len(getattr(annotation_type, "__args__", ())) > 0:\n'
        '            annotation_type = typing.List[annotation_type.__args__[0]]\n'
        '        if annotation_type not in SUPPORTED_PARAM_TYPES.keys():'
    )

    if target in content:
        content = content.replace(target, replacement, 1)
        with open(p, "w") as f:
            f.write(content)
        print("✅ PyTorch infer_schema patched successfully!")
    else:
        print("ℹ️ Target pattern not found or already patched.")

if __name__ == "__main__":
    patch_torch()
