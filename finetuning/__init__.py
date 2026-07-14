# finetuning/ — LLM fine-tuning experiment suite for the Explainer agent.
#
# Task: teach Mistral-7B-Instruct to turn raw SHAP/LIME attribution output
# into clear plain-English explanations for non-technical users.
#
# Run order:
#   1. python -m finetuning.dataset_prep          (build + optimize the dataset)
#   2. python -m finetuning.finetune_lora         (or qlora / dora / dpo / orpo / galore)
#   3. python -m finetuning.benchmark             (compare all trained adapters)
#   4. python -m finetuning.export_ollama         (merge best adapter -> GGUF -> Ollama)
