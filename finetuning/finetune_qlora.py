# finetuning/finetune_qlora.py

"""
Experiment 2 — QLoRA (4-bit quantized base).

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

BACKENDS
  --backend unsloth       (default) Unsloth's fused Triton kernels — ~2x
                          faster, ~50% less VRAM. Linux/WSL2 recommended.
  --backend transformers  Plain HF transformers + bitsandbytes + PEFT.
                          Works everywhere (incl. native Windows) and with
                          any HF causal-LM via --base-model. Used for
                          small-GPU smoke runs (e.g. Qwen2.5-0.5B on 4 GB).

EXPERIMENT TRACKING
  --report-to mlflow (default) logs losses/hyperparams to ./mlruns.
  View with:  mlflow ui --port 5000

Usage:
    python -m finetuning.finetune_qlora --epochs 3 --rank 16
    # 4 GB-GPU smoke run (real training, small base — label results as such):
    python -m finetuning.finetune_qlora --backend transformers \
        --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 1
"""

import argparse
import os

from finetuning.common import (
    BASE_MODEL_HF, BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, SEED,
    RunTracker, format_example, get_final_losses, load_sft_splits, require_cuda,
)

# LoRA recipe shared by both backends — the experiment variable between
# LoRA/QLoRA/DoRA is NEVER these values:
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune (4-bit base)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--backend", choices=["unsloth", "transformers"],
                        default="unsloth")
    parser.add_argument("--base-model", type=str, default=None,
                        help="HF model id override (default: Mistral-7B-Instruct)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training examples (smoke runs)")
    parser.add_argument("--report-to", type=str, default="mlflow",
                        choices=["mlflow", "none"],
                        help="Experiment tracker (mlflow logs to ./mlruns)")
    args = parser.parse_args()

    require_cuda("qlora")
    output_dir = os.path.join(MODELS_DIR, "qlora")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "datapilot-finetuning")

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── 1. Load 4-bit base via the chosen backend ─────────────────────────
    if args.backend == "unsloth":
        base_model = args.base_model or BASE_MODEL_UNSLOTH_4BIT
        print(f"Loading {base_model} (4-bit, Unsloth)...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            use_gradient_checkpointing="unsloth", random_state=SEED,
        )
    else:
        base_model = args.base_model or BASE_MODEL_HF
        print(f"Loading {base_model} (4-bit NF4, transformers+bitsandbytes)...")
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    # ── 2. Data — identical splits + prompt format for every experiment ───
    train_rows, val_rows, _ = load_sft_splits()
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
        val_rows = val_rows[:max(8, args.max_samples // 8)]
    to_text = lambda rows: Dataset.from_dict({
        "text": [format_example(r["input"], r["output"], tokenizer.eos_token)
                 for r in rows]
    })
    train_ds, val_ds = to_text(train_rows), to_text(val_rows)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | base: {base_model}")

    # ── 3. Train ──────────────────────────────────────────────────────────
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
            save_strategy="no",   # final save below; avoids double checkpoints
            bf16=True,
            optim="paged_adamw_8bit",
            seed=SEED,
            report_to=args.report_to if args.report_to != "none" else "none",
            run_name=f"qlora-{base_model.split('/')[-1]}",
        ),
    )
    trainer.train()

    train_loss, eval_loss = get_final_losses(trainer)
    tracker.stop(final_loss=train_loss, eval_loss=eval_loss, extra={
        "base_model": base_model,
        "method": f"QLoRA (4-bit NF4 base, {args.backend})",
        "backend": args.backend,
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "epochs": args.epochs, "n_train": len(train_ds),
    })

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save()
    print(f"\nQLoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
