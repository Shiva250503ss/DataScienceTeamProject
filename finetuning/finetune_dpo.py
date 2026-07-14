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

WHY IT FITS THIS TASK
  "A good explanation" is a PREFERENCE, not a single gold string. Two
  explanations can both be fine; what we really know is that grounded plain
  English beats jargon. DPO encodes exactly that.

PIPELINE ORDER
  Standard practice is SFT first, then DPO on top. This script therefore
  loads the QLoRA adapter from finetune_qlora.py if it exists (recommended),
  or falls back to the raw instruct base.

Usage:
    python -m finetuning.finetune_qlora        # (recommended) SFT first
    python -m finetuning.finetune_dpo --epochs 1 --beta 0.1
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_prompt, get_final_losses, load_preference_splits,
    require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="DPO preference tuning (TRL + Unsloth)")
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
                        help="Path to an SFT adapter to start from (QLoRA default)")
    args = parser.parse_args()

    require_cuda("dpo")
    output_dir = os.path.join(MODELS_DIR, "dpo")

    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()  # patches TRL's DPOTrainer with Unsloth fast kernels

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    # ── 1. Load model: SFT adapter if available, else raw base ────────────
    start_from = args.from_adapter if os.path.exists(
        os.path.join(args.from_adapter, "adapter_config.json")) else BASE_MODEL_UNSLOTH_4BIT
    if start_from == args.from_adapter:
        print(f"Starting DPO from SFT adapter: {start_from} (recommended path)")
    else:
        print("! No SFT adapter found — starting DPO from the raw instruct base. "
              "For best results run finetune_qlora.py first.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=start_from,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # If we started from the raw base, we still need trainable adapters:
    if start_from == BASE_MODEL_UNSLOTH_4BIT:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

    # ── 2. Preference data ────────────────────────────────────────────────
    # TRL's DPOTrainer expects columns: prompt / chosen / rejected.
    # The prompt gets the same [INST] wrapper used everywhere else.
    train_rows, val_rows = load_preference_splits()
    to_ds = lambda rows: Dataset.from_dict({
        "prompt": [format_prompt(r["prompt"]) for r in rows],
        "chosen": [" " + r["chosen"] for r in rows],
        "rejected": [" " + r["rejected"] for r in rows],
    })
    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
    print(f"Preference pairs — train: {len(train_ds)} | val: {len(val_ds)}")

    # ── 3. Train ──────────────────────────────────────────────────────────
    tracker = RunTracker("dpo", output_dir)
    tracker.start()

    trainer = DPOTrainer(
        model=model,
        # ref_model=None + PEFT adapters: TRL disables the adapters to
        # compute reference logprobs — no second 7B copy in memory.
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
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,
            optim="paged_adamw_8bit",
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_SEQ_LENGTH - 512,  # leave room for responses
            seed=SEED,
            report_to="none",
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": start_from,
        "method": "DPO (preference pairs, TRL + Unsloth)",
        "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
        "started_from_sft_adapter": start_from == args.from_adapter,
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nDPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
