#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick smoke test for a merged HF model (after LoRA merge, PhoGPT-4B-Chat LoRA, v.v.)

Usage examples:
  # Cách 1: gõ prompt ngay trong shell, có \n sẽ được chuyển thành xuống dòng thật
  python test_model.py \
    --model ./merged_model_v2_chat \
    --prompt "### CÂU HỎI: Ô tô là gì?\n\n### TRẢ LỜI:\n" \
    --max-new-tokens 200

  # Cách 2: không truyền --prompt, gõ trực tiếp trong stdin (giữ format y chang data train)
  python test_model.py --model ./merged_model_v2_chat --max-new-tokens 200
  (sau đó gõ:)
  ### CÂU HỎI: Ô tô là gì?

  ### TRẢ LỜI:
  (Ctrl+D để kết thúc)
"""

from __future__ import annotations
import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_prompt_text(raw: str) -> str:
    """
    Chuyển các chuỗi '\\n' literal thành newline thật.
    Ví dụ: 'A\\n\\nB' -> 'A\n\nB'
    Giúp prompt giống với format đã dùng khi fine-tune trong JSON.
    """
    if "\\n" in raw:
        return raw.replace("\\n", "\n")
    return raw


def build_inputs(tokenizer, prompt: str):
    """
    Build model inputs cho mô hình decoder-only, dùng raw text.
    (Không dùng chat_template vì bạn fine-tune theo format ### CÂU HỎI / ### TRẢ LỜI.)
    """
    return tokenizer(prompt, return_tensors="pt")


def main():
    ap = argparse.ArgumentParser(description="Test merged HF model (CPU)")
    ap.add_argument("--model", required=True, help="Path hoặc HF id của model đã merge LoRA")
    ap.add_argument("--prompt", help="User prompt/question. Nếu bỏ trống, đọc từ stdin")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy (no sampling)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-fix-newlines",
        action="store_true",
        help="Không tự động chuyển '\\n' literal thành newline thật trong --prompt",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"\n📦 Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    print(f"🤖 Loading model (CPU, fp32): {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    model.eval()
    model.to("cpu")

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Cố gắng sync eos_token_id giữa tokenizer & model config
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    gen_cfg = getattr(model, "generation_config", None)
    print("\n🔧 Special tokens / EOS alignment:")
    print(f"  tokenizer.eos_token        = {repr(tokenizer.eos_token)}")
    print(f"  tokenizer.eos_token_id     = {tokenizer.eos_token_id}")
    print(f"  tokenizer.pad_token        = {repr(tokenizer.pad_token)}")
    print(f"  tokenizer.pad_token_id     = {tokenizer.pad_token_id}")
    print(f"  model.config.eos_token_id  = {getattr(model.config, 'eos_token_id', None)}")
    if gen_cfg is not None:
        print(f"  generation_config.eos_id   = {getattr(gen_cfg, 'eos_token_id', None)}")
    else:
        print("  generation_config          = None")

    # Lấy prompt
    prompt = args.prompt
    if not prompt:
        print("\nNhập prompt (Ctrl-D để kết thúc): ", file=sys.stderr)
        prompt = sys.stdin.read().strip()
        if not prompt:
            # fallback demo
            prompt = (
                "### CÂU HỎI: Hệ thống phanh trên ô tô có công dụng chính là gì?\n\n"
                "### TRẢ LỜI:\n"
            )
    else:
        if not args.no_fix_newlines:
            prompt = normalize_prompt_text(prompt)

    print("\n==== PROMPT (input text) ====")
    print(prompt)

    inputs = build_inputs(tokenizer, prompt)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    print(f"\n🔎 Input shape: {tuple(input_ids.shape)} (batch, seq_len)")

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": max(1e-5, args.temperature),
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    print("\n🚀 Generating...\n")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    generated = outputs[0]
    prompt_len = input_ids.shape[1]

    print(f"🔎 Generated shape: {tuple(generated.shape)} (total tokens)")
    print(f"   Prompt length (tokens): {prompt_len}")
    print(f"   New tokens count       : {generated.shape[0] - prompt_len}")
    print(f"🔢 New token IDs          : {generated[prompt_len:].tolist()}")

    # FULL decode (prompt + output), no skipping specials
    full_decoded = tokenizer.decode(generated, skip_special_tokens=False)

    # Only new tokens (no skip special)
    new_tokens_raw = tokenizer.decode(generated[prompt_len:], skip_special_tokens=False)

    # Only new tokens (skip special)
    new_tokens_clean = tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)

    print("\n==== FULL DECODE (prompt + output, NO skip_special_tokens) ====")
    print(full_decoded)

    print("\n==== ONLY NEW TOKENS (NO skip_special_tokens) ====")
    print(new_tokens_raw)

    print("\n==== ONLY NEW TOKENS (skip_special_tokens=True) ====")
    print(new_tokens_clean.strip())


if __name__ == "__main__":
    main()
