# finetuning/finetune_dpo.py

"""
Experiment 4 — DPO (Direct Preference Optimization) via TRL.

WHAT DPO DOES (Rafailov et al., 2023)
  Instead of "imitate this target text" (SFT), DPO trains on PAIRS:
  for the same SHAP input we have a CHOSEN explanation (clear, grounded,
  jargon-free) and a REJECTED one (jargon dump / vague filler / factually
  wrong driver). The loss pushes the policy to assign higher likelihood to
  chosen over rejected, RELATIVE to a frozen reference model:

      L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                               - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

  It is RLHF's objective solved in closed form — no reward model, no PPO
  rollouts. beta controls how far the policy may drift from the reference.

PIPELINE ORDER
  Standard practice is SFT first, then DPO on top. This script loads the
  QLoRA adapter from finetune_qlora.py if present (recommended) or starts
  from the raw base.

BACKENDS — same convention as finetune_qlora.py:
  --backend unsloth       (default) fused kernels, Linux/WSL2 recommended
  --backend transformers  plain HF + bitsandbytes + PEFT, any model/OS
                          (small-GPU smoke runs with --base-model)

Usage:
    python -m finetuning.finetune_qlora            # (recommended) SFT first
    python -m finetuning.finetune_dpo --epochs 1 --beta 0.1
    # 4 GB-GPU smoke run:
    python -m finetuning.finetune_dpo --backend transformers \
        --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 1
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_HF, BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_prompt, get_final_losses, load_preference_splits,
    require_cuda,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser(description="DPO preference tuning (TRL)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="DPO overfits fast — 1 epoch is usually right")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty strength: lower = drift further from reference")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Much lower than SFT — DPO gradients are sharp")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--from-adapter", type=str,
                        default=os.path.join(MODELS_DIR, "qlora"),
                        help="SFT adapter to start from (QLoRA default)")
    parser.add_argument("--backend", choices=["unsloth", "transformers"],
                        default="unsloth")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap preference pairs (smoke runs)")
    parser.add_argument("--report-to", type=str, default="mlflow",
                        choices=["mlflow", "none"])
    args = parser.parse_args()

    require_cuda("dpo")
    output_dir = os.path.join(MODELS_DIR, "dpo")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "datapilot-finetuning")

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    has_sft_adapter = os.path.exists(
        os.path.join(args.from_adapter, "adapter_config.json"))

    # ── Preference data FIRST (before any CUDA model exists) ──────────────
    # Building the pyarrow Datasets after loading the quantized model +
    # injecting adapter weights hard-crashes (0xC0000005) on Windows with
    # pyarrow 24 + torch 2.6 + bitsandbytes 0.49. Creating the arrow tables
    # before the model touches the GPU sidesteps the DLL/state interaction.
    train_rows, val_rows = load_preference_splits()
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
        val_rows = val_rows[:max(8, args.max_samples // 8)]
    to_ds = lambda rows: Dataset.from_dict({
        "prompt": [format_prompt(r["prompt"]) for r in rows],
        "chosen": [" " + r["chosen"] for r in rows],
        "rejected": [" " + r["rejected"] for r in rows],
    })
    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
    print(f"Preference pairs — train: {len(train_ds)} | val: {len(val_ds)}",
          flush=True)

    # ── 1. Load model: SFT adapter if available, else raw base ────────────
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()
        start_from = args.from_adapter if has_sft_adapter else \
            (args.base_model or BASE_MODEL_UNSLOTH_4BIT)
        print(f"Starting DPO from: {start_from}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=start_from,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        if not has_sft_adapter:
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32, lora_dropout=0.0,
                target_modules=LORA_TARGETS, bias="none",
                use_gradient_checkpointing="unsloth", random_state=SEED,
            )
    else:
        import torch
        from peft import (LoraConfig, get_peft_model,
                          prepare_model_for_kbit_training,
                          set_peft_model_state_dict)
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        base_model = args.base_model or BASE_MODEL_HF
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token_id is None:
            # trl 0.9.x DPOTrainer prepends bos_token_id unconditionally;
            # Qwen2 tokenizers have no BOS -> torch.tensor([None, ...]) crash.
            tokenizer.bos_token = tokenizer.eos_token
        model = prepare_model_for_kbit_training(model)
        # Always wrap with a fresh LoRA config, then inject the SFT adapter
        # weights via state dict. (PeftModel.from_pretrained(is_trainable=True)
        # on a 4-bit base hard-crashes with an access violation on Windows —
        # peft 0.12 + bitsandbytes 0.49; the state-dict route is equivalent
        # and stable.)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            task_type="CAUSAL_LM",
        ))
        if has_sft_adapter:
            print(f"Starting DPO from SFT adapter: {args.from_adapter}", flush=True)
            from safetensors.torch import load_file
            sd = load_file(os.path.join(args.from_adapter,
                                        "adapter_model.safetensors"))
            print("  adapter safetensors loaded", flush=True)
            load_result = set_peft_model_state_dict(model, sd)
            print("  adapter weights injected", flush=True)
            if load_result.unexpected_keys:
                raise RuntimeError(f"Adapter mismatch: {load_result.unexpected_keys[:5]}")
        else:
            print("! No SFT adapter found — DPO from the raw base. "
                  "Run finetune_qlora.py first for best results.")
        start_from = args.from_adapter if has_sft_adapter else base_model

    # ── 3. Train ──────────────────────────────────────────────────────────
    tracker = RunTracker("dpo", output_dir)
    tracker.start()

    trainer = DPOTrainer(
        model=model,
        # ref_model=None + PEFT: TRL disables the adapters to compute
        # reference logprobs — no second full model copy in memory.
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=DPOConfig(
            output_dir=output_dir,
            beta=args.beta,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            bf16=True,
            optim="paged_adamw_8bit",
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_SEQ_LENGTH - 512,
            seed=SEED,
            report_to=args.report_to if args.report_to != "none" else "none",
            run_name=f"dpo-{start_from.split('/')[-1].split(os.sep)[-1]}",
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": start_from,
        "method": f"DPO (preference pairs, TRL, {args.backend})",
        "backend": args.backend,
        "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
        "started_from_sft_adapter": has_sft_adapter,
        "n_train_pairs": len(train_ds),
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nDPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
