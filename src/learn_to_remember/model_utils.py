# Origin: ported from GRPO-Zero
#   (https://github.com/Vivek-Chaudhari30/GRPO-Zero, main @ ce51ed9):
#   - src/model_utils.py: load_config, get_device, resolve_dtype, load_tokenizer,
#     load_model_and_tokenizer, load_model_with_adapter, load_policy_for_training
#   - src/train.py: set_seed, save_checkpoint, maybe_resume
#   Changed: LoRA wrapping is factored out into `apply_lora` (so tests can wrap a
#   tiny offline model), checkpoints also store RNG state, set_seed also seeds
#   python/numpy, and `from_pretrained` uses the dtype kwarg the installed
#   transformers version expects.
"""Model, LoRA, device and checkpoint helpers shared by eval and training.

Device policy: cuda -> mps -> cpu. dtype policy ("auto"): bf16 on cuda, fp16 on
mps (8 GB friendly), fp32 on cpu. Everything can be overridden from config.
Note: the T4 has no native bf16 (it is emulated and slow), so set
`model.dtype: fp16` there.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import transformers
import yaml

STATE_FILE = "training_state.pt"   # optimizer + step + history + RNG, for --resume

_DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------- #
# Config, seeding, device
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dt = _DTYPE_MAP.get(str(name).lower(), None)
    if dt is not None:
        return dt
    # "auto"
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """transformers renamed `torch_dtype` to `dtype` (4.56). Use whichever the
    installed version expects, so Colab/Kaggle images with older versions work."""
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_tokenizer(model_id: str):
    """Tokenizer configured for left-padded batched generation (decoder-only)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model_and_tokenizer(model_id: str, *, dtype_name: str = "auto", device=None):
    from transformers import AutoModelForCausalLM

    device = device or get_device()
    dtype = resolve_dtype(dtype_name, device)
    tok = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **_dtype_kwarg(dtype))
    model.to(device)
    model.eval()
    return model, tok, device


def load_model_with_adapter(model_id: str, adapter_path: str, *, dtype_name: str = "auto",
                            device=None):
    """Load the base model plus a trained LoRA adapter, for evaluating a trained policy."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = device or get_device()
    dtype = resolve_dtype(dtype_name, device)
    tok = load_tokenizer(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, **_dtype_kwarg(dtype))
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return model, tok, device


def apply_lora(model, lora_cfg: dict, *, grad_checkpoint: bool = False):
    """Wrap a causal LM in a LoRA adapter. The frozen *reference* policy is the
    same object with the adapter disabled (see grpo.core.batched_logprobs), so we
    keep one copy of the base weights plus a small trainable adapter, not two
    full models."""
    from peft import LoraConfig, get_peft_model

    # generation passes use_cache=True explicitly; keep the cache off for the grad
    # forwards (also required for gradient checkpointing).
    model.config.use_cache = False
    peft_cfg = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    if grad_checkpoint:
        # Recompute layer activations in the backward instead of storing them all,
        # the biggest memory lever here. With a frozen base + LoRA, the inputs must
        # require grad so the recomputed graph reconnects through the adapter.
        model.enable_input_require_grads()
        # transformers 5 renamed this; older versions only have the old name.
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
    return model


def load_policy_for_training(model_id: str, *, lora_cfg: dict, dtype_name: str = "auto",
                             device=None, grad_checkpoint: bool = False):
    from transformers import AutoModelForCausalLM

    device = device or get_device()
    dtype = resolve_dtype(dtype_name, device)
    tok = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **_dtype_kwarg(dtype))
    model = apply_lora(model, lora_cfg, grad_checkpoint=grad_checkpoint)
    model.to(device)
    return model, tok, device


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #
def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=device) may have moved these to the GPU; RNG
    # setters require CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(save_dir, model, optimizer, step, history):
    """Save the LoRA adapter (for eval) plus resumable training state, so a run
    cut off by a GPU-quota disconnect can pick up where it left off."""
    from peft import get_peft_model_state_dict

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)  # adapter_model.safetensors, used by eval --adapter
    torch.save(
        {"step": step, "model": get_peft_model_state_dict(model),
         "optimizer": optimizer.state_dict(), "history": history, "rng": _rng_state()},
        os.path.join(save_dir, STATE_FILE),
    )


def maybe_resume(save_dir, model, optimizer, device):
    """If save_dir/training_state.pt exists, restore adapter + optimizer + history
    (+ RNG state) and return (start_step, history). Otherwise return (0, [])."""
    from peft import set_peft_model_state_dict

    path = os.path.join(save_dir, STATE_FILE)
    if not os.path.exists(path):
        return 0, []
    ckpt = torch.load(path, map_location=device, weights_only=False)
    set_peft_model_state_dict(model, ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "rng" in ckpt:
        _set_rng_state(ckpt["rng"])
    start = ckpt["step"] + 1
    print(f"Resuming from {path}: continuing at step {start}")
    return start, ckpt["history"]
