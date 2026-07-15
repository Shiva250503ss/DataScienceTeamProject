# finetuning/finetune_orpo.py

"""
Experiment 5 — ORPO (Odds Ratio Preference Optimization) via TRL.

WHAT ORPO DOES (Hong et al., 2024)
  ORPO merges SFT and preference alignment into ONE training stage:

      L = L_SFT(chosen) + lambda * L_OR

  where L_OR penalizes the odds ratio of generating the rejected response
  relative to the chosen one. Two practical consequences:
    1. NO reference model — unlike DPO there is no frozen copy to keep in
       memory and no KL anchor; the odds-ratio term itself prevents collapse.
    2. NO separate SFT stage — the chosen responses ARE the SFT targets.

DPO vs ORPO IN THIS SUITE (the interview comparison)
  - DPO:  two stages (SFT -> DPO), needs reference logprobs, beta-tuned.
  - ORPO: one stage from the raw base, ~half the total compute, one knob
          (lambda). Trades a bit of controllability for simplicity.
  Both train on the SAME preference pairs from dataset_prep.py, so
  benchmark.py gives a clean head-to-head.

Usage:
    python -m finetuning.finetune_orpo --epochs 2 --orpo-lambda 0.1
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_prompt, get_final_losses, load_preference_splits,
    require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="ORPO one-stage preference tuning (TRL + Unsloth)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="ORPO doubles as SFT, so it tolerates more epochs than DPO")
    parser.add_argument("--orpo-lambda", type=float, default=0.1,
                        help="Weight of the odds-ratio penalty vs the SFT loss")
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    require_cuda("orpo")
    output_dir = os.path.join(MODELS_DIR, "orpo")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import ORPOConfig, ORPOTrainer

    # ── 1. Raw base — ORPO's whole point is skipping the SFT stage ────────
    print(f"Loading {BASE_MODEL_UNSLOTH_4BIT} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_UNSLOTH_4BIT,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # ── 2. Same preference pairs as DPO ───────────────────────────────────
    train_rows, val_rows = load_preference_splits()
    to_ds = lambda rows: Dataset.from_dict({
        "prompt": [format_prompt(r["prompt"]) for r in rows],
        "chosen": [" " + r["chosen"] for r in rows],
        "rejected": [" " + r["rejected"] for r in rows],
    })
    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
    print(f"Preference pairs — train: {len(train_ds)} | val: {len(val_ds)}")

    # ── 3. Train ──────────────────────────────────────────────────────────
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "datapilot-finetuning")
    tracker = RunTracker("orpo", output_dir)
    tracker.start()

    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=ORPOConfig(
            output_dir=output_dir,
            beta=args.orpo_lambda,   # TRL names ORPO's lambda 'beta'
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,
            optim="paged_adamw_8bit",
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_SEQ_LENGTH - 512,
            seed=SEED,
            report_to=os.getenv("FT_REPORT_TO", "mlflow"),
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": BASE_MODEL_UNSLOTH_4BIT,
        "method": "ORPO (one-stage SFT + preference, TRL + Unsloth)",
        "orpo_lambda": args.orpo_lambda, "lr": args.lr, "epochs": args.epochs,
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nORPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
