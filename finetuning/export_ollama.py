# finetuning/export_ollama.py

"""
Export the best adapter to Ollama: merge -> GGUF -> Modelfile -> `ollama create`.

WHY THIS STEP EXISTS
  Training produces a LoRA adapter (safetensors) that needs Python + GPU to
  run. The production platform talks to Ollama. This script closes the loop:

    1. MERGE    — fold the adapter into the base weights (W' = W + BA),
                  producing a standalone fp16 model
    2. GGUF     — convert to llama.cpp's GGUF format with q4_k_m quantization
                  (~4.4 GB file, runs on CPU or any GPU, no Python needed)
    3. Modelfile— declares the Mistral chat template + our system prompt so
                  the tuned behavior survives inside Ollama
    4. CREATE   — registers the model as `datapilot-explainer` in Ollama

  After this, agents/base.py just needs OLLAMA_MODEL=datapilot-explainer in
  .env — zero code changes, no API key, no token limits.

PREREQUISITE for step 2: a llama.cpp checkout (for convert_hf_to_gguf.py):
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements.txt

Usage:
    python -m finetuning.export_ollama --technique qlora
    python -m finetuning.export_ollama --technique dpo --quant q5_k_m
"""

import argparse
import os
import subprocess
import sys

from finetuning.common import (
    BASE_MODEL_HF, MODELS_DIR, SYSTEM_INSTRUCTION, require_cuda,
)

EXPORT_DIR = os.path.join(MODELS_DIR, "export")


def merge_adapter(technique: str, backend: str = "unsloth") -> str:
    """
    Step 1 — Load base in fp16, apply the adapter, merge (W' = W + BA),
    save the standalone full model.

    backend="unsloth":       save_pretrained_merged on the Mistral-7B recipe.
    backend="transformers":  plain PEFT merge_and_unload — reads the
                             adapter's own base_model_name_or_path, so smoke
                             adapters (e.g. Qwen2.5-0.5B) merge against the
                             exact base they were trained on.
    """
    import json as _json

    adapter_path = os.path.join(MODELS_DIR, technique)
    adapter_cfg = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        # GaLore saves a full checkpoint, not an adapter — no merge needed
        if os.path.exists(os.path.join(adapter_path, "config.json")):
            print(f"[merge] {technique} is a full checkpoint — skipping merge.")
            return adapter_path
        raise SystemExit(f"No adapter found at {adapter_path}. Train it first.")

    merged_dir = os.path.join(EXPORT_DIR, f"{technique}_merged")
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"[merge] Reusing existing merged model at {merged_dir}")
        return merged_dir

    if backend == "unsloth":
        print(f"[merge] Merging {technique} adapter into {BASE_MODEL_HF} (fp16, unsloth)...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,  # loads adapter on 4-bit base; merge is fp16
        )
        model.save_pretrained_merged(merged_dir, tokenizer,
                                     save_method="merged_16bit")
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base = _json.load(f).get("base_model_name_or_path", BASE_MODEL_HF)
        print(f"[merge] Merging {technique} adapter into {base} (fp16, transformers)...")
        # fp16 base (NOT quantized) — merge must happen in full precision,
        # and the 4-bit PeftModel crash does not affect the fp16 path.
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map="cpu")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    print(f"[merge] Merged fp16 model saved to {merged_dir}")
    return merged_dir


def convert_to_gguf(merged_dir: str, technique: str, quant: str,
                    llama_cpp_dir: str) -> str:
    """
    Step 2 — HF safetensors -> GGUF via llama.cpp's converter, then quantize.
    q4_k_m is the sweet spot: ~4.4 GB, negligible quality loss for a 7B.
    """
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise SystemExit(
            f"convert_hf_to_gguf.py not found in {llama_cpp_dir}.\n"
            f"Clone llama.cpp first:  git clone https://github.com/ggerganov/llama.cpp"
        )

    os.makedirs(EXPORT_DIR, exist_ok=True)
    f16_gguf = os.path.join(EXPORT_DIR, f"{technique}_f16.gguf")
    final_gguf = os.path.join(EXPORT_DIR, f"datapilot-explainer_{quant}.gguf")

    print(f"[gguf] Converting {merged_dir} -> {f16_gguf} ...")
    subprocess.run([sys.executable, convert_script, merged_dir,
                    "--outfile", f16_gguf, "--outtype", "f16"], check=True)

    # llama-quantize binary lives in the llama.cpp build dir; try both names
    quant_bin = None
    for candidate in ("llama-quantize", "llama-quantize.exe", "quantize"):
        for sub in ("", "build/bin"):
            p = os.path.join(llama_cpp_dir, sub, candidate)
            if os.path.exists(p):
                quant_bin = p
                break
        if quant_bin:
            break

    if quant_bin:
        print(f"[gguf] Quantizing to {quant} ...")
        subprocess.run([quant_bin, f16_gguf, final_gguf, quant], check=True)
        os.remove(f16_gguf)  # 14 GB intermediate — delete once quantized
    else:
        print("[gguf] ! llama-quantize binary not found (llama.cpp not built). "
              "Using the f16 GGUF directly — works, but 14 GB instead of ~4.4 GB.")
        final_gguf = f16_gguf

    print(f"[gguf] Done: {final_gguf}")
    return final_gguf


def write_modelfile(gguf_path: str) -> str:
    """
    Step 3 — Modelfile: tells Ollama the weights, the Mistral [INST] chat
    template, and bakes in the Explainer system prompt so callers don't have
    to resend it. num_ctx matches the training sequence length.
    """
    modelfile_path = os.path.join(EXPORT_DIR, "Modelfile")
    content = f'''# DataPilot Explainer — fine-tuned Mistral-7B-Instruct
# Turns SHAP/LIME attribution output into plain-English explanations.
FROM {os.path.abspath(gguf_path)}

TEMPLATE """[INST] {{{{ if .System }}}}{{{{ .System }}}}

{{{{ end }}}}{{{{ .Prompt }}}} [/INST]"""

SYSTEM """{SYSTEM_INSTRUCTION}"""

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[modelfile] Written to {modelfile_path}")
    return modelfile_path


def create_in_ollama(modelfile_path: str, model_name: str):
    """Step 4 — register with the local Ollama server."""
    print(f"[ollama] Creating model '{model_name}' ...")
    # Resolve the ollama binary: PATH first, then OLLAMA_EXE env var
    # (portable installs, e.g. %USERPROFILE%\ollama-portable\ollama.exe).
    ollama_bin = os.getenv("OLLAMA_EXE", "ollama")
    try:
        subprocess.run([ollama_bin, "create", model_name, "-f", modelfile_path],
                       check=True)
    except FileNotFoundError:
        raise SystemExit(
            "The `ollama` CLI was not found. Install Ollama (or set OLLAMA_EXE "
            "to its full path), then run:\n"
            f"  ollama create {model_name} -f {modelfile_path}"
        )
    print(f"""
[done] Model registered. Try it:

    ollama run {model_name} "Prediction: will churn (confidence: 84%) ..."

To make the platform use it, set in .env:

    OLLAMA_MODEL={model_name}
""")


def main():
    parser = argparse.ArgumentParser(description="Merge best adapter and export to Ollama")
    parser.add_argument("--technique", type=str, required=True,
                        help="Which trained model to export (e.g. qlora, dpo) — "
                             "pick the winner from benchmark_results.md")
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0"],
                        help="GGUF quantization level")
    parser.add_argument("--llama-cpp-dir", type=str, default="./llama.cpp",
                        help="Path to a llama.cpp checkout")
    parser.add_argument("--model-name", type=str, default="datapilot-explainer",
                        help="Name to register in Ollama")
    parser.add_argument("--backend", choices=["unsloth", "transformers"],
                        default="unsloth",
                        help="Merge backend (transformers for smoke adapters; "
                             "runs on CPU, no GPU required)")
    args = parser.parse_args()

    if args.backend == "unsloth":
        require_cuda("export")  # unsloth merge needs GPU; GGUF conversion is CPU

    merged = merge_adapter(args.technique, backend=args.backend)
    gguf = convert_to_gguf(merged, args.technique, args.quant, args.llama_cpp_dir)
    modelfile = write_modelfile(gguf)
    create_in_ollama(modelfile, args.model_name)


if __name__ == "__main__":
    main()
