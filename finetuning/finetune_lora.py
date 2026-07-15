# finetuning/finetune_lora.py

"""
Experiment 1 — Standard LoRA (16-bit base) via Unsloth.

WHAT LoRA DOES
  Instead of updating all 7.2B weights, LoRA freezes the base model and
  injects small trainable low-rank matrices (A: d x r, B: r x d) beside each
  targeted weight matrix W. The effective weight becomes W + (alpha/r) * B @ A.
  With r=16 on Mistral-7B that is ~0.6% of parameters — the adapter file is
  ~80 MB instead of a 14 GB checkpoint.

WHY UNSLOTH
  Unsloth rewrites the attention/MLP backward passes with custom Triton
  kernels: ~2x faster training and ~50% less VRAM than vanilla PEFT, with
  numerically identical results. Same LoraConfig semantics underneath.

VRAM: ~18 GB (16-bit base + activations). If that's too much for your GPU,
use finetune_qlora.py (4-bit base, ~7 GB).

Usage:
    python -m finetuning.finetune_lora --epochs 3 --rank 16
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_HF, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_example, get_final_losses, load_sft_splits, require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune (Unsloth, 16-bit base)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha (scale = alpha/r)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    require_cuda("lora")
    output_dir = os.path.join(MODELS_DIR, "lora")

    # Unsloth must be imported BEFORE transformers/trl to patch the kernels
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── 1. Load 16-bit base model ─────────────────────────────────────────
    print(f"Loading {BASE_MODEL_HF} in 16-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_HF,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,   # <- the ONLY difference vs finetune_qlora.py
        dtype=None,           # auto: bfloat16 on Ampere+, float16 otherwise
    )

    # ── 2. Attach LoRA adapters ───────────────────────────────────────────
    # target_modules covers every linear layer in Mistral's attention (q,k,v,o)
    # and MLP (gate, up, down) — the standard "all-linear" recipe that
    # consistently beats attention-only LoRA on instruction tasks.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,          # 0 enables Unsloth's fast path
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # recompute activations: big VRAM save
        random_state=SEED,
    )

    # ── 3. Data: same splits + prompt format as every other experiment ───
    train_rows, val_rows, _ = load_sft_splits()
    to_text = lambda rows: Dataset.from_dict({
        "text": [format_example(r["input"], r["output"], tokenizer.eos_token)
                 for r in rows]
    })
    train_ds, val_ds = to_text(train_rows), to_text(val_rows)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 4. Train ──────────────────────────────────────────────────────────
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "datapilot-finetuning")
    tracker = RunTracker("lora", output_dir)
    tracker.start()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,  # effective batch = 16
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,                # Ampere+; flips to fp16 automatically below
            optim="adamw_8bit",       # 8-bit optimizer states halve optimizer VRAM
            seed=SEED,
            report_to=os.getenv("FT_REPORT_TO", "mlflow"),
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": BASE_MODEL_HF, "method": "LoRA (16-bit base, Unsloth)",
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "epochs": args.epochs,
        "trainable_params_pct": round(
            100 * sum(p.numel() for p in model.parameters() if p.requires_grad)
            / sum(p.numel() for p in model.parameters()), 3),
    })

    # ── 5. Save adapter only (small) — merging happens in export_ollama.py ─
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nLoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
