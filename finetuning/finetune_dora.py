# finetuning/finetune_dora.py

"""
Experiment 3 — DoRA (Weight-Decomposed Low-Rank Adaptation) via PEFT.

WHAT DoRA CHANGES vs LoRA (Liu et al., 2024)
  LoRA learns W' = W + BA — a single low-rank delta on the whole weight.
  DoRA first decomposes each weight into magnitude and direction:
      W = m * (V / ||V||)
  then applies the low-rank update ONLY to the direction V while learning
  the magnitude vector m separately. This mirrors how full fine-tuning
  actually moves weights (mostly directional changes with independent
  magnitude scaling) and typically recovers 1-2 points of quality vs LoRA
  at the SAME rank — at the cost of ~20-30% slower training.

WHY PEFT (not Unsloth) HERE
  DoRA is a one-flag change in HuggingFace PEFT (use_dora=True in
  LoraConfig), which makes this script a clean side-by-side against the
  Unsloth LoRA/QLoRA runs: same data, same prompt format, same rank.
  The base is loaded in 4-bit (bitsandbytes) so it fits consumer GPUs.

Usage:
    python -m finetuning.finetune_dora --epochs 3 --rank 16
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_HF, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_example, get_final_losses, load_sft_splits, require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="DoRA fine-tune (PEFT, 4-bit base)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    require_cuda("dora")
    output_dir = os.path.join(MODELS_DIR, "dora")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments)
    from trl import SFTTrainer

    # ── 1. Load base in 4-bit NF4 (same quantization recipe as QLoRA) ─────
    print(f"Loading {BASE_MODEL_HF} in 4-bit NF4 via bitsandbytes...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_HF,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_HF)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token

    # kbit prep: casts layernorms to fp32, enables gradient checkpointing,
    # and makes the input embeddings require grads — required for stable
    # 4-bit training with PEFT.
    model = prepare_model_for_kbit_training(model)

    # ── 2. DoRA config — identical to the LoRA runs except use_dora=True ──
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,   # <- THE experiment: magnitude/direction decomposition
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── 3. Data ───────────────────────────────────────────────────────────
    train_rows, val_rows, _ = load_sft_splits()
    to_text = lambda rows: Dataset.from_dict({
        "text": [format_example(r["input"], r["output"], tokenizer.eos_token)
                 for r in rows]
    })
    train_ds, val_ds = to_text(train_rows), to_text(val_rows)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 4. Train ──────────────────────────────────────────────────────────
    tracker = RunTracker("dora", output_dir)
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
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            seed=SEED,
            report_to="none",
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": BASE_MODEL_HF,
        "method": "DoRA (weight-decomposed LoRA, PEFT, 4-bit base)",
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "epochs": args.epochs,
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nDoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
