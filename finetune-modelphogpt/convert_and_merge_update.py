#!/usr/bin/env python3
"""
Cải tiến của `convert_and_merge.py`.
Tính năng thêm:
 - logging thay cho print
 - CLI flags để override r, lora_alpha, lora_dropout, target_modules
 - detect metadata trong checkpoint nếu có (tìm 'lora_config' hoặc 'peft_config')
 - dry-run mode để chỉ báo cáo mapping keys mà không ghi model
 - báo cáo chi tiết missing/unexpected keys khi load_state_dict (strict=False)
 - cố gắng sửa `config.json` auto_map cho tokenizer giống `merge_lora.py`
 - nhiều try/except thân thiện

Usage example:
  python convert_and_merge_update.py --checkpoint ./out_checkpoints/latest-rank0.pt \
      --base vinai/PhoGPT-4B-Chat --output ./merged_model_update --dry-run

"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def setup_logger(verbosity: int):
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=level,
    )


def infer_lora_metadata(checkpoint: Dict) -> Dict:
    """Try to locate Lora metadata (r, lora_alpha, lora_dropout, target_modules)
    inside the checkpoint if present. Returns a dict with any found keys.
    This is heuristic and best-effort.
    """
    meta = {}
    # common keys that might appear in custom checkpoints
    for key in ["lora_config", "peft_config", "adapter_config", "lora_state"]:
        if key in checkpoint:
            v = checkpoint[key]
            if isinstance(v, dict):
                meta.update({k: v.get(k) for k in ["r", "lora_alpha", "lora_dropout", "target_modules"] if k in v})
    # Some trainers store config under 'state' or nested 'state' structures
    if "state" in checkpoint and isinstance(checkpoint["state"], dict):
        for sub in [checkpoint["state"], checkpoint["state"].get("model", {})]:
            if isinstance(sub, dict):
                for key in ["lora_config", "peft_config"]:
                    if key in sub and isinstance(sub[key], dict):
                        meta.update({k: sub[key].get(k) for k in ["r", "lora_alpha", "lora_dropout", "target_modules"] if k in sub[key]})
    return meta


def find_lora_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k in state_dict.keys() if any(x in k for x in ["lora_A", "lora_B", "base_layer", "lora.weight"]) ]


def clean_key_prefix(key: str) -> str:
    for prefix in ["model.", "_forward_module.", "module."]:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def save_auto_map_tokenizer(out_dir: str):
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            logging.debug("Không thể đọc config.json để update auto_map")
            return
        auto_map = cfg.get("auto_map", {}) or {}
        # Only set AutoTokenizer if not set or points to something else that is known-broken
        if auto_map.get("AutoTokenizer") != "GPTNeoXTokenizerFast":
            auto_map["AutoTokenizer"] = "GPTNeoXTokenizerFast"
            cfg["auto_map"] = auto_map
            try:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                logging.info("Updated config.json auto_map.AutoTokenizer -> GPTNeoXTokenizerFast")
            except Exception:
                logging.warning("Không thể ghi config.json để cập nhật auto_map")


def convert_and_merge_lora(
    checkpoint_path: str,
    base_model: str,
    output_dir: str,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[List[str]] = None,
    dry_run: bool = False,
    trust_remote_code: bool = False,
):
    logging.info("Start convert_and_merge_lora")

    # 1. Load checkpoint
    logging.info("Loading checkpoint: %s", checkpoint_path)
    try:
        # Most Composer checkpoints include non-tensor objects and require
        # weights_only=False to fully unpickle. This may execute code from
        # the checkpoint so only use for trusted files.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logging.debug("Checkpoint top-level keys: %s", list(checkpoint.keys())[:10])
    except Exception as e:
        logging.error("Failed to load checkpoint: %s", e)
        return False

    # 2. Extract state dict
    state_dict = None
    if isinstance(checkpoint, dict):
        if "state" in checkpoint and isinstance(checkpoint["state"], dict):
            # Many checkpoints store model under state['model']
            if "model" in checkpoint["state"]:
                state_dict = checkpoint["state"]["model"]
        if state_dict is None and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        if state_dict is None:
            # maybe checkpoint itself is a state_dict
            state_dict = checkpoint
    else:
        logging.error("Unsupported checkpoint format")
        return False

    if not isinstance(state_dict, dict):
        logging.error("Unable to find state_dict in checkpoint")
        return False

    logging.info("State dict contains %d keys", len(state_dict))

    # find lora keys
    lora_keys = find_lora_keys(state_dict)
    logging.info("Found %d LoRA keys", len(lora_keys))
    if len(lora_keys) == 0:
        logging.warning("No LoRA keys detected in checkpoint; aborting")
        return False

    # 3. Infer metadata if present
    meta = infer_lora_metadata(checkpoint)
    logging.debug("Inferred metadata from checkpoint: %s", meta)

    # Apply CLI overrides or fallback to inferred metadata
    r_val = r or meta.get("r")
    alpha_val = lora_alpha or meta.get("lora_alpha")
    dropout_val = lora_dropout if lora_dropout is not None else meta.get("lora_dropout")

    # guess target modules if not provided
    detected_str = " ".join(lora_keys)
    if target_modules is None:
        if any(t in detected_str for t in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "Wqkv" in detected_str:
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    logging.info("Using Lora params: r=%s, lora_alpha=%s, lora_dropout=%s", r_val, alpha_val, dropout_val)
    logging.info("Target modules: %s", target_modules)

    # 4. Load base model with conservative settings
    logging.info("Loading base model: %s", base_model)
    try:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        try:
            setattr(config, "init_device", "cpu")
        except Exception:
            pass
        try:
            if hasattr(config, "attn_config") and isinstance(config.attn_config, dict):
                config.attn_config["attn_impl"] = "torch"
        except Exception:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            # attn_implementation may not be supported by all models; ignore if error
            # We'll attempt to pass it and catch errors
            **({"attn_implementation": "eager"} if True else {}),
        )
    except TypeError:
        # fallback if attn_implementation not supported
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logging.error("Failed to load base model: %s", e)
        return False

    logging.info("Base model loaded")

    # 5. Prepare PEFT wrapper and load LoRA weights
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        logging.error("peft package required but not found: %s", e)
        return False

    lora_cfg_kwargs = {
        "r": int(r_val) if r_val is not None else 16,
        "lora_alpha": int(alpha_val) if alpha_val is not None else 32,
        "lora_dropout": float(dropout_val) if dropout_val is not None else 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
    }

    logging.debug("LoraConfig kwargs: %s", lora_cfg_kwargs)

    lora_config = LoraConfig(**lora_cfg_kwargs)
    peft_model = get_peft_model(model, lora_config)

    # Build peft_state_dict mapping cleaned keys -> tensors
    peft_state_dict = {}
    for key in lora_keys:
        clean_key = clean_key_prefix(key)
        peft_state_dict[clean_key] = state_dict[key]

    logging.info("Prepared peft_state_dict with %d keys", len(peft_state_dict))

    # Before loading: report some examples
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        sample_keys = list(peft_state_dict.keys())[:20]
        logging.debug("Sample peft keys: %s", sample_keys)

    # Attempt to load and collect missing/unexpected
    try:
        load_res = peft_model.load_state_dict(peft_state_dict, strict=False)
        # load_state_dict may return a NamedTuple with missing_keys/unexpected_keys (PyTorch)
        missing = getattr(load_res, "missing_keys", None)
        unexpected = getattr(load_res, "unexpected_keys", None)
        if missing is None and unexpected is None:
            # older behaviour may raise warnings instead; try to inspect by comparing keys
            model_keys = set(peft_model.state_dict().keys())
            provided = set(peft_state_dict.keys())
            missing = sorted(list(model_keys - provided))[:50]
            unexpected = sorted(list(provided - model_keys))[:50]
    except Exception as e:
        logging.exception("Error while loading LoRA state dict into PEFT model: %s", e)
        return False

    logging.info("Load complete. missing_keys=%d, unexpected_keys=%d", len(missing or []), len(unexpected or []))
    if missing:
        logging.debug("Missing keys (sample 50): %s", missing[:50])
    if unexpected:
        logging.debug("Unexpected keys in provided weights (sample 50): %s", unexpected[:50])

    if dry_run:
        logging.info("Dry-run enabled, exiting before merge/save")
        return True

    # Merge and unload
    try:
        logging.info("Merging LoRA into base model...")
        merged = peft_model.merge_and_unload()
    except Exception as e:
        logging.exception("Failed to merge and unload: %s", e)
        return False

    # 6. Save merged model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    try:
        logging.info("Saving merged model to %s", output_dir)
        merged.save_pretrained(output_dir, safe_serialization=True)

        # Save tokenizer from base
        try:
            tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=trust_remote_code)
            tok.save_pretrained(output_dir)
        except Exception:
            logging.warning("Failed to save tokenizer from base model; continuing")

        # Attempt to update auto_map for tokenizer class (for GPTNeoX-like models)
        save_auto_map_tokenizer(output_dir)

    except Exception as e:
        logging.exception("Failed to save merged model: %s", e)
        return False

    logging.info("Saved merged model successfully")
    # Print output folder summary
    try:
        for f in os.listdir(output_dir):
            size = os.path.getsize(os.path.join(output_dir, f)) / (1024 ** 2)
            logging.info(" - %s (%.1f MB)", f, size)
    except Exception:
        pass

    return True


def parse_args():
    p = argparse.ArgumentParser(description="Convert Composer checkpoint and merge LoRA (improved)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--r", type=int, help="LoRA rank r (overrides checkpoint if provided)")
    p.add_argument("--lora_alpha", type=int, help="LoRA alpha (overrides checkpoint if provided)")
    p.add_argument("--lora_dropout", type=float, help="LoRA dropout (overrides checkpoint if provided)")
    p.add_argument("--target_modules", type=str, help="Comma separated list of target modules to apply LoRA to")
    p.add_argument("--dry-run", action="store_true", help="Do everything except saving merged model")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading models/configs")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (use -v or -vv)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.verbose)
    target_modules = None
    if args.target_modules:
        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]

    ok = convert_and_merge_lora(
        checkpoint_path=args.checkpoint,
        base_model=args.base,
        output_dir=args.output,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        dry_run=args.dry_run,
        trust_remote_code=args.trust_remote_code,
    )
    if not ok:
        logging.error("Conversion failed")
        raise SystemExit(1)
    logging.info("Conversion finished successfully")


if __name__ == "__main__":
    main()
