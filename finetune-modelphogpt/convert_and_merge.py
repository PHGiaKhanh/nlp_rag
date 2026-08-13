#!/usr/bin/env python3
"""
Script để convert Composer checkpoint sang HF format và merge LoRA vào base model

Cách dùng:
    python convert_and_merge.py --checkpoint ./out_checkpoints/latest-rank0.pt --output ./merged_model
"""

import argparse
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def convert_and_merge_lora(checkpoint_path: str, base_model: str, output_dir: str):
    """
    Convert Composer checkpoint và merge LoRA vào base model
    
    Args:
        checkpoint_path: Đường dẫn tới checkpoint Composer (.pt)
        base_model: Tên/đường dẫn base model (ví dụ: vinai/PhoGPT-4B)
        output_dir: Thư mục output để lưu merged model
    """
    
    print("="*70)
    print("🚀 CONVERT & MERGE LORA CHECKPOINT")
    print("="*70)
    
    # 1. Load checkpoint
    print(f"\n📦 Đang load checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Đã load checkpoint")
        print(f"   Keys: {list(checkpoint.keys())[:5]}...")
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return False
    
    # 2. Load base model (tắt FlashAttention, ép chạy CPU/eager)
    print(f"\n🤖 Đang load base model: {base_model}")
    try:
        # Một số model (PhoGPT-4B-Chat) dùng remote code và có hỗ trợ FlashAttention.
        # Trên CPU sẽ lỗi flash_attn_triton.py. Chúng ta ép attention thuần (eager/torch).
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
        # Ép chạy trên CPU
        try:
            setattr(config, 'init_device', 'cpu')
        except Exception:
            pass
        # Nếu model có attn_config theo kiểu MPT, đặt attn_impl='torch'
        try:
            if hasattr(config, 'attn_config') and isinstance(config.attn_config, dict):
                config.attn_config['attn_impl'] = 'torch'
                # Các cờ khác an toàn mặc định
        except Exception:
            pass
        
        # attn_implementation='eager' sẽ tránh FA2 nếu model hỗ trợ tham số này
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        print(f"✅ Đã load base model")
    except Exception as e:
        print(f"❌ Lỗi load base model: {e}")
        return False
    
    # 3. Extract state dict
    print(f"\n🔧 Đang extract state dict từ checkpoint...")
    try:
        if 'state' in checkpoint:
            state_dict = checkpoint['state']['model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"✅ Tìm thấy {len(state_dict)} keys trong state dict")
        
        # Lọc chỉ lấy LoRA weights (base_layer, lora_A, lora_B)
        lora_keys = [k for k in state_dict.keys() if any(x in k for x in ['lora_A', 'lora_B', 'base_layer'])]
        print(f"✅ Tìm thấy {len(lora_keys)} LoRA parameters")
        
        if len(lora_keys) == 0:
            print("⚠️  Không tìm thấy LoRA parameters trong checkpoint!")
            print("   Checkpoint này có thể không phải là LoRA checkpoint")
            return False
        
    except Exception as e:
        print(f"❌ Lỗi extract state dict: {e}")
        return False
    
    # 4. Load PEFT model và merge
    print(f"\n🔗 Đang merge LoRA vào base model...")
    try:
        from peft import PeftModel, LoraConfig, get_peft_model
        
        # Suy đoán tập target_modules theo keys trong checkpoint để tránh mismatch
        # - LLaMA-like: q_proj/k_proj/v_proj/o_proj + MLP gate/up/down
        # - MPT-like: Wqkv/out_proj + up/down (không có gate_proj)
        detected_keys = " ".join(lora_keys)
        if any(t in detected_keys for t in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "Wqkv" in detected_keys:
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        else:
            # fallback an toàn: giữ tập phổ biến của LLaMA
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Tạo LoRA config (cần phù hợp với lúc training)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        
        # Wrap model với PEFT
        model = get_peft_model(model, lora_config)
        
        # Load LoRA weights từ checkpoint
        # Cần mapping keys từ Composer format sang PEFT format
        peft_state_dict = {}
        for key in lora_keys:
            # Remove prefix nếu có (ví dụ: 'model.', '_forward_module.')
            clean_key = key
            for prefix in ['model.', '_forward_module.', 'module.']:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            
            peft_state_dict[clean_key] = state_dict[key]
        
        # Load weights
        model.load_state_dict(peft_state_dict, strict=False)
        print(f"✅ Đã load LoRA weights")
        
        # Merge và unload
        print(f"🔀 Đang merge LoRA vào base model...")
        model = model.merge_and_unload()
        print(f"✅ Đã merge thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi merge LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Save merged model
    print(f"\n💾 Đang lưu merged model vào: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model.save_pretrained(output_dir, safe_serialization=True)
        
        # Save tokenizer (không cần remote code)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Đã lưu merged model!")
        print(f"\n📂 Nội dung thư mục output:")
        for f in os.listdir(output_dir):
            size = os.path.getsize(os.path.join(output_dir, f)) / (1024**2)
            print(f"   - {f} ({size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Lỗi lưu model: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"Model đã được merge và lưu tại: {output_dir}")
    print(f"Bạn có thể dùng model này trực tiếp với transformers:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}")')
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert Composer checkpoint và merge LoRA")
    parser.add_argument("--checkpoint", type=str, default="./out_checkpoints/latest-rank0.pt",
                        help="Đường dẫn tới Composer checkpoint (.pt)")
    parser.add_argument("--base", type=str, default="vinai/PhoGPT-4B-Chat",
                        help="Base model name/path")
    parser.add_argument("--output", type=str, default="./merged_model",
                        help="Output directory cho merged model")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint không tồn tại: {args.checkpoint}")
        return
    
    # Run conversion
    success = convert_and_merge_lora(
        checkpoint_path=args.checkpoint,
        base_model=args.base,
        output_dir=args.output
    )
    
    if not success:
        print("\n❌ Conversion thất bại!")
        exit(1)


if __name__ == "__main__":
    main()
