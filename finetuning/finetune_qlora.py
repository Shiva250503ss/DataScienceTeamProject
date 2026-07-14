# finetuning/finetune_qlora.py

"""
Experiment 2 — QLoRA (4-bit quantized base) via Unsloth.

WHAT QLoRA ADDS OVER LoRA
  The frozen base model is stored in 4-bit NF4 (NormalFloat4) instead of
  16-bit. Three tricks from the QLoRA paper (Dettmers et al., 2023):
    1. NF4 quantization — information-theoretically optimal for normally
       distributed weights (better than plain int4)
    2. Double quantization — the quantization constants themselves are
       quantized, saving another ~0.4 bits/param
    3. Paged optimizers — optimizer states page to CPU RAM on VRAM spikes
  The LoRA adapters still train in 16-bit, so gradient quality is preserved;
  only the frozen weights are compressed.

WHY THIS IS THE HEADLINE EXPERIMENT
  ~7 GB VRAM → trains on a single consumer GPU (RTX 3060/4060). Typical
  quality drop vs 16-bit LoRA on a task like ours: ≈1% — benchmark.py
  measures that gap directly.

Usage:
    python -m finetuning.finetune_qlora --epochs 3 --rank 16
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_example, get_final_losses, load_sft_splits, require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune (Unsloth, 4-bit base)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    require_cuda("qlora")
    output_dir = os.path.join(MODELS_DIR, "qlora")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── 1. Load pre-quantized 4-bit base ──────────────────────────────────
    # Unsloth's -bnb-4bit repo ships weights already in NF4, so loading takes
    # ~5 GB of download instead of 14 GB and no quantization pass is needed.
    print(f"Loading {BASE_MODEL_UNSLOTH_4BIT} (4-bit NF4)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_UNSLOTH_4BIT,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,    # <- QLoRA switch
        dtype=None,
    )

    # ── 2. LoRA adapters — identical hyperparams to finetune_lora.py so the
    #       ONLY variable between the two experiments is base precision. ───
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # ── 3. Data ───────────────────────────────────────────────────────────
    train_rows, val_rows, _ = load_sft_splits()
    to_text = lambda rows: Dataset.from_dict({
        "text": [format_example(r["input"], r["output"], tokenizer.eos_token)
                 for r in rows]
    })
    train_ds, val_ds = to_text(train_rows), to_text(val_rows)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 4. Train ──────────────────────────────────────────────────────────
    tracker = RunTracker("qlora", output_dir)
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
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,
            # Paged optimizer = QLoRA paper's trick #3: spills optimizer
            # state to CPU RAM instead of OOM-ing on VRAM spikes.
            optim="paged_adamw_8bit",
            seed=SEED,
            report_to="none",
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": BASE_MODEL_UNSLOTH_4BIT,
        "method": "QLoRA (4-bit NF4 base, Unsloth)",
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "epochs": args.epochs,
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nQLoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
