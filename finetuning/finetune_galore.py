# finetuning/finetune_galore.py

"""
Experiment 6 (OPTIONAL / EXPERIMENTAL) — GaLore full-parameter training.

>>> Marked experimental: run the LoRA-family experiments first. This one
>>> needs ~24 GB VRAM and galore-torch, and is included to demonstrate the
>>> memory/quality frontier BEYOND adapters — skip if time-constrained. <<<

WHAT GaLore DOES (Zhao et al., 2024 — Gradient Low-Rank Projection)
  LoRA constrains the WEIGHT update to low rank. GaLore instead observes
  that the GRADIENT matrix is naturally low-rank during training, so it:
    1. computes the full gradient G for each weight matrix
    2. projects it into a low-rank subspace (P^T G, via periodic SVD)
    3. keeps Adam's optimizer states (m, v) ONLY in that small subspace
    4. projects the update back up and applies it to the FULL weights
  Result: every parameter still moves (true full fine-tuning expressivity),
  but optimizer memory drops from 2x model size to a fraction — 7B full
  training fits in ~24 GB instead of ~60 GB with plain AdamW.

TRADE-OFFS vs LoRA/QLoRA
  + No adapter rank ceiling — can fit things adapters can't
  - Slower (periodic SVD of gradients), more VRAM than QLoRA
  - Produces a FULL 14 GB checkpoint, not an 80 MB adapter

Usage:
    pip install galore-torch
    python -m finetuning.finetune_galore --epochs 1
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_HF, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_example, get_final_losses, load_sft_splits, require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description="GaLore full-parameter fine-tune (experimental)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Full-parameter training moves fast — 1 epoch default")
    parser.add_argument("--galore-rank", type=int, default=128,
                        help="Rank of the gradient projection subspace")
    parser.add_argument("--update-proj-gap", type=int, default=200,
                        help="Steps between SVD refreshes of the projector")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Full-param LR — 20x lower than LoRA's 2e-4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    require_cuda("galore")
    try:
        import galore_torch  # noqa: F401
    except ImportError:
        raise SystemExit("galore-torch not installed. Run: pip install galore-torch")

    output_dir = os.path.join(MODELS_DIR, "galore")

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments)
    from trl import SFTTrainer

    # ── 1. Full-precision base (bf16) — GaLore trains ALL weights, so the
    #       model cannot be quantized like in the adapter experiments. ─────
    print(f"Loading {BASE_MODEL_HF} in bf16 (full weights trainable)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_HF,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_HF)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Data — identical to every other experiment ─────────────────────
    train_rows, val_rows, _ = load_sft_splits()
    to_text = lambda rows: Dataset.from_dict({
        "text": [format_example(r["input"], r["output"], tokenizer.eos_token)
                 for r in rows]
    })
    train_ds, val_ds = to_text(train_rows), to_text(val_rows)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 3. Train with the GaLore optimizer ────────────────────────────────
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "datapilot-finetuning")
    tracker = RunTracker("galore", output_dir)
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
            # HF Transformers has native GaLore support: the optimizer name
            # plus optim_target_modules routes the projected optimizer to
            # the attention/MLP matrices (layerwise = lowest memory variant).
            optim="galore_adamw_8bit_layerwise",
            optim_target_modules=["attn", "mlp"],
            optim_args=(f"rank={args.galore_rank}, "
                        f"update_proj_gap={args.update_proj_gap}, scale=0.25"),
            gradient_checkpointing=True,
            seed=SEED,
            report_to=os.getenv("FT_REPORT_TO", "mlflow"),
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": BASE_MODEL_HF,
        "method": "GaLore full-parameter (gradient low-rank projection)",
        "galore_rank": args.galore_rank, "lr": args.lr, "epochs": args.epochs,
        "note": "full checkpoint, not an adapter",
    })

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nGaLore full checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
