# finetuning/dataset_prep.py

"""
Training Dataset Builder + Optimizer for the Explainer fine-tune.

Produces two dataset families in finetuning/data/:

  1. SFT data (train/val/test.jsonl) — {"input": <SHAP summary>, "output": <good explanation>}
     Used by: LoRA, QLoRA, DoRA, GaLore, and as the SFT part of ORPO.

  2. Preference data (preferences_train/val.jsonl) —
     {"prompt": <SHAP summary>, "chosen": <good explanation>, "rejected": <bad explanation>}
     Used by: DPO, ORPO.

Data sources:
  a) SYNTHETIC — programmatically generated SHAP scenarios across 8 business
     domains (churn, credit, housing, medical, HR, sales, fraud, marketing).
     Good/bad explanations are written by local Mistral via Ollama when it is
     running; otherwise a deterministic template writer is used, so this
     script ALWAYS runs without any API key or server.
  b) REAL — explanation narratives persisted by past ExplainerAgent runs
     (output*/explanations/local_narratives.json — written by agents/explainer.py).

Dataset OPTIMIZATION steps (the part interviewers ask about):
  1. Exact + near-duplicate removal (normalized text hashing + Jaccard shingles)
  2. Quality filtering — length bounds, jargon leakage, degenerate repetition,
     explanation must reference at least one input feature
  3. Class balancing — equal coverage of prediction directions and domains
     (downsample the majority, so no synthetic oversampling artifacts)
  4. Before/after statistics report -> finetuning/data/dataset_report.md

Usage:
    python -m finetuning.dataset_prep --n-synthetic 800 --use-ollama
    python -m finetuning.dataset_prep --n-synthetic 400          # template mode, no LLM
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import requests

from finetuning.common import (
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE,
    PREF_TRAIN_FILE, PREF_VAL_FILE, SEED, save_jsonl,
)

random.seed(SEED)

# ── Ollama access (optional — template fallback keeps the script self-contained) ──
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ollama_generate(prompt: str, temperature: float = 0.7) -> Optional[str]:
    """One-shot generation via local Ollama. Returns None on any failure."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": 256}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC SHAP SCENARIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
# Each domain defines realistic features with value ranges. A scenario picks
# 4-6 features, assigns them values and signed SHAP attributions, and states
# the model's prediction. This mirrors EXACTLY what agents/explainer.py feeds
# to the LLM at runtime (see LOCAL_EXPLANATION_PROMPT there).

DOMAINS = {
    "customer_churn": {
        "task": "classification",
        "predictions": ["will churn", "will stay"],
        "features": {
            "monthly_charges": (20, 120, "$"), "tenure_months": (1, 72, " months"),
            "contract_type_encoded": (0, 2, ""), "support_tickets": (0, 12, " tickets"),
            "total_charges": (100, 8000, "$"), "internet_service_fiber": (0, 1, ""),
            "payment_autopay": (0, 1, ""), "streaming_services": (0, 4, ""),
        },
    },
    "credit_risk": {
        "task": "classification",
        "predictions": ["high risk", "low risk"],
        "features": {
            "annual_income": (18000, 200000, "$"), "debt_to_income_ratio": (0.05, 0.95, ""),
            "credit_history_years": (0, 30, " years"), "num_late_payments": (0, 15, ""),
            "loan_amount": (1000, 50000, "$"), "employment_years": (0, 25, " years"),
            "credit_utilization": (0.0, 1.0, ""), "num_open_accounts": (1, 20, ""),
        },
    },
    "house_price": {
        "task": "regression",
        "predictions": ["${:,.0f}"],
        "features": {
            "square_feet": (600, 5000, " sq ft"), "bedrooms": (1, 6, ""),
            "bathrooms": (1, 4, ""), "year_built": (1930, 2024, ""),
            "distance_to_city_km": (0.5, 45, " km"), "school_rating": (1, 10, "/10"),
            "lot_size_acres": (0.05, 2.5, " acres"), "garage_spaces": (0, 3, ""),
        },
    },
    "medical_readmission": {
        "task": "classification",
        "predictions": ["likely readmission", "unlikely readmission"],
        "features": {
            "patient_age": (18, 95, " years"), "num_prior_admissions": (0, 10, ""),
            "num_medications": (0, 25, ""), "length_of_stay_days": (1, 30, " days"),
            "num_lab_procedures": (1, 80, ""), "has_diabetes": (0, 1, ""),
            "num_diagnoses": (1, 12, ""), "emergency_admission": (0, 1, ""),
        },
    },
    "employee_attrition": {
        "task": "classification",
        "predictions": ["will leave", "will stay"],
        "features": {
            "years_at_company": (0, 30, " years"), "monthly_salary": (2500, 20000, "$"),
            "job_satisfaction": (1, 5, "/5"), "overtime_hours_month": (0, 60, " hours"),
            "years_since_promotion": (0, 12, " years"), "commute_distance_km": (1, 60, " km"),
            "training_sessions_year": (0, 8, ""), "manager_rating": (1, 5, "/5"),
        },
    },
    "sales_forecast": {
        "task": "regression",
        "predictions": ["${:,.0f}"],
        "features": {
            "marketing_spend": (500, 100000, "$"), "store_size_sqm": (80, 3000, " sqm"),
            "competitor_distance_km": (0.1, 20, " km"), "avg_foot_traffic": (50, 5000, " visitors/day"),
            "promo_active": (0, 1, ""), "holiday_week": (0, 1, ""),
            "staff_count": (2, 60, ""), "years_open": (0, 40, " years"),
        },
    },
    "fraud_detection": {
        "task": "classification",
        "predictions": ["fraudulent", "legitimate"],
        "features": {
            "transaction_amount": (1, 15000, "$"), "hour_of_day": (0, 23, "h"),
            "distance_from_home_km": (0, 8000, " km"), "merchant_risk_score": (0.0, 1.0, ""),
            "transactions_last_hour": (0, 20, ""), "card_present": (0, 1, ""),
            "account_age_days": (1, 5000, " days"), "amount_vs_avg_ratio": (0.1, 50.0, "x"),
        },
    },
    "marketing_conversion": {
        "task": "classification",
        "predictions": ["will convert", "won't convert"],
        "features": {
            "pages_visited": (1, 40, ""), "session_duration_min": (0.5, 60, " min"),
            "previous_purchases": (0, 30, ""), "email_opens_month": (0, 25, ""),
            "days_since_last_visit": (0, 180, " days"), "cart_value": (0, 1200, "$"),
            "is_mobile": (0, 1, ""), "referral_channel_paid": (0, 1, ""),
        },
    },
}


def _fmt_value(val: float, lo, hi, unit: str) -> str:
    """Format a feature value naturally (ints stay ints, $ goes in front)."""
    if isinstance(lo, int) and isinstance(hi, int) and unit != "":
        val = round(val)
    if unit == "$":
        return f"${val:,.0f}"
    if isinstance(lo, int) and isinstance(hi, int):
        return f"{round(val)}{unit}"
    return f"{val:.2f}{unit}"


def generate_scenario(domain_name: str) -> Dict:
    """
    Generate one synthetic SHAP scenario:
      - picks a prediction (or regression value)
      - picks 4-6 features with random values
      - assigns each a signed SHAP attribution; the sum is consistent with
        the prediction direction so examples are logically coherent
    Returns dict with 'shap_input' (text block) and metadata for balancing.
    """
    spec = DOMAINS[domain_name]
    features = random.sample(list(spec["features"].items()), k=random.randint(4, 6))

    if spec["task"] == "classification":
        prediction = random.choice(spec["predictions"])
        pred_positive = prediction == spec["predictions"][0]
        confidence = round(random.uniform(0.62, 0.97), 2)
        pred_line = f"Prediction: {prediction} (confidence: {confidence:.0%})"
    else:
        value = random.uniform(80_000, 900_000) if domain_name == "house_price" \
            else random.uniform(5_000, 400_000)
        prediction = spec["predictions"][0].format(value)
        pred_positive = True
        pred_line = f"Prediction: {prediction}"

    # Signed SHAP values: majority must push toward the prediction
    lines, feats_meta = [], []
    n_supporting = max(2, len(features) - random.randint(1, 2))
    for i, (fname, (lo, hi, unit)) in enumerate(features):
        val = random.uniform(lo, hi)
        supports = i < n_supporting
        magnitude = round(random.uniform(0.05, 0.9) * (1.5 if i == 0 else 1.0), 3)
        sign = 1 if (supports == pred_positive) else -1
        shap_val = sign * magnitude
        direction = "increases" if shap_val > 0 else "decreases"
        lines.append(
            f"  - {fname} = {_fmt_value(val, lo, hi, unit)} -> {direction} "
            f"prediction (SHAP: {shap_val:+.3f})"
        )
        feats_meta.append(fname)

    random.shuffle(lines)
    shap_input = (
        f"Domain: {domain_name.replace('_', ' ')}\n"
        f"{pred_line}\n"
        f"Top factors behind this prediction (SHAP values):\n" + "\n".join(lines)
    )
    return {
        "shap_input": shap_input,
        "domain": domain_name,
        "prediction_class": prediction,
        "features": feats_meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLANATION WRITERS (good + bad)
# ═══════════════════════════════════════════════════════════════════════════════

GOOD_WRITER_PROMPT = """Below is the feature-attribution output of an ML model.
Write a 2-4 sentence explanation for a non-technical business user.
Rules: plain English only; never say 'SHAP', 'LIME', 'feature', 'attribution',
or any statistics term; name the concrete factors and their real values;
end with what would most likely change the outcome.

{shap_input}

Explanation:"""

BAD_WRITER_PROMPT = """Below is the feature-attribution output of an ML model.
Write a DELIBERATELY POOR explanation of this prediction — the kind a lazy
data scientist would write. Make it one of: (a) full of jargon like 'the SHAP
value of X was +0.4', (b) vague and generic without naming any actual factor,
or (c) confidently wrong about which factor mattered most. 2-3 sentences.

{shap_input}

Poor explanation:"""


def template_good_explanation(scenario: Dict) -> str:
    """
    Deterministic fallback writer — parses the scenario's own SHAP lines and
    verbalizes them. Quality is decent because the structure is fixed; it lets
    the whole suite run end-to-end with no LLM server.
    """
    lines = re.findall(
        r"- (\w+) = ([^\s].*?) -> (increases|decreases) prediction \(SHAP: ([+-][\d.]+)\)",
        scenario["shap_input"],
    )
    lines.sort(key=lambda t: abs(float(t[3])), reverse=True)
    pred = scenario["prediction_class"]
    top = lines[0]
    # "Supporting" = same SHAP direction as the strongest driver;
    # "opposing" = the opposite direction. Defining it relative to the top
    # driver keeps the story coherent whichever class was predicted.
    second = next((l for l in lines[1:] if l[2] == top[2]), None)
    opposing = next((l for l in lines[1:] if l[2] != top[2]), None)

    def nice(name):  # snake_case -> words
        return name.replace("_", " ")

    text = (f"The model predicted '{pred}' mainly because {nice(top[0])} "
            f"is {top[1]}, which strongly pushed the decision in this direction.")
    if second is not None:
        text += (f" {nice(second[0]).capitalize()} at {second[1]} "
                 f"added further weight to the same conclusion.")
    if opposing is not None:
        text += (f" On the other hand, {nice(opposing[0])} ({opposing[1]}) "
                 f"worked against this outcome, but not strongly enough.")
    text += (f" Changing {nice(top[0])} would have the biggest effect "
             f"on this prediction.")
    return text


def template_bad_explanation(scenario: Dict) -> str:
    """Deterministic 'rejected' writer — produces one of three failure modes."""
    mode = random.choice(["jargon", "vague", "wrong"])
    lines = re.findall(
        r"- (\w+) = ([^\s].*?) -> (increases|decreases) prediction \(SHAP: ([+-][\d.]+)\)",
        scenario["shap_input"],
    )
    pred = scenario["prediction_class"]
    if mode == "jargon":
        parts = [f"{n} had a SHAP attribution of {s}" for n, v, d, s in lines[:3]]
        return (f"The model output '{pred}' because " + ", ".join(parts) +
                ". The additive feature attributions sum to the logit delta "
                "from the base value.")
    if mode == "vague":
        return ("The model looked at several factors in the data and decided "
                f"'{pred}' was the most likely outcome. Many variables "
                "contributed to this. The prediction is based on patterns "
                "found during training.")
    # wrong — claim the LEAST important feature was the main driver
    lines.sort(key=lambda t: abs(float(t[3])))
    least = lines[0]
    return (f"The prediction '{pred}' is almost entirely driven by "
            f"{least[0].replace('_', ' ')} being {least[1]}. "
            f"No other factor played a meaningful role in this decision.")


def write_explanations(scenario: Dict, use_ollama: bool) -> Tuple[str, str]:
    """Return (good, bad) explanation pair for one scenario."""
    good = bad = None
    if use_ollama:
        good = ollama_generate(GOOD_WRITER_PROMPT.format(shap_input=scenario["shap_input"]),
                               temperature=0.7)
        bad = ollama_generate(BAD_WRITER_PROMPT.format(shap_input=scenario["shap_input"]),
                              temperature=0.9)
    if not good:
        good = template_good_explanation(scenario)
    if not bad:
        bad = template_bad_explanation(scenario)
    return good.strip(), bad.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REAL EXAMPLES — harvested from past ExplainerAgent runs
# ═══════════════════════════════════════════════════════════════════════════════

def harvest_real_examples() -> List[Dict]:
    """
    Scan output*/explanations/local_narratives.json files written by
    agents/explainer.py (each run persists its LLM narratives + the SHAP
    context that produced them). Older runs that only saved HTML charts
    contain no usable text and are skipped.
    """
    examples = []
    pattern = os.path.join(os.path.dirname(DATA_DIR), "..", "output*",
                           "explanations", "local_narratives.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                narratives = json.load(f)
            for n in narratives:
                shap_input = n.get("shap_context") or ""
                explanation = n.get("narrative") or ""
                if shap_input and explanation and len(explanation) > 60:
                    examples.append({
                        "shap_input": shap_input,
                        "good": explanation,
                        "domain": "real_run",
                        "prediction_class": n.get("prediction", "unknown"),
                    })
        except Exception:
            continue
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATASET OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _shingles(text: str, k: int = 5) -> set:
    words = _normalize(text).split()
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def dedupe(rows: List[Dict], jaccard_threshold: float = 0.7) -> Tuple[List[Dict], int]:
    """
    Two-stage duplicate removal on the INPUT side:
      1. exact duplicates via md5 of normalized text — O(n)
      2. near-duplicates via 5-gram Jaccard similarity — O(n²) but fine at
         this dataset size (~1k examples); catches re-rolls of the same
         scenario with only value changes
    """
    seen_hashes, kept, removed = set(), [], 0
    for row in rows:
        h = hashlib.md5(_normalize(row["shap_input"]).encode()).hexdigest()
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        kept.append(row)

    final, shingle_cache = [], []
    for row in kept:
        sh = _shingles(row["shap_input"])
        is_dup = False
        for other in shingle_cache:
            inter = len(sh & other)
            union = len(sh | other) or 1
            if inter / union > jaccard_threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            final.append(row)
            shingle_cache.append(sh)
    return final, removed


JARGON_WORDS = ("shap", "lime", "attribution", "logit", "coefficient",
                "feature importance", "z-score", "p-value")


def quality_filter(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Drop low-quality GOOD explanations. A pair survives only if the 'good'
    side actually satisfies the task contract:
      - length between 120 and 900 characters (2-4 real sentences)
      - no leaked jargon words (the whole point of the fine-tune)
      - not degenerate (no 4x repetition of the same 4-gram)
      - mentions at least one input feature by (partial) name — proves the
        explanation is grounded in the input, not generic filler
    """
    reasons = Counter()
    kept = []
    for row in rows:
        good = row["good"]
        gl = good.lower()
        if not (120 <= len(good) <= 900):
            reasons["bad_length"] += 1
            continue
        if any(j in gl for j in JARGON_WORDS):
            reasons["jargon_leak"] += 1
            continue
        grams = re.findall(r"(?=((?:\w+ ){3}\w+))", _normalize(good))
        if grams and Counter(grams).most_common(1)[0][1] >= 4:
            reasons["degenerate_repetition"] += 1
            continue
        feature_names = re.findall(r"- (\w+) =", row["shap_input"])
        mentioned = any(
            part in gl
            for f in feature_names for part in [f.replace("_", " "), f.split("_")[0]]
            if len(part) > 3
        )
        if feature_names and not mentioned:
            reasons["ungrounded"] += 1
            continue
        kept.append(row)
    return kept, dict(reasons)


def balance(rows: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Balance across (domain, prediction-direction) buckets by downsampling to
    1.5x the median bucket size. Downsampling (not duplication) keeps the
    dataset honest — no example appears twice.
    """
    buckets = defaultdict(list)
    for row in rows:
        # Group regression predictions into one bucket per domain
        pred = row["prediction_class"]
        pred_key = "regression_value" if pred.startswith("$") else pred
        buckets[(row["domain"], pred_key)].append(row)

    sizes = sorted(len(v) for v in buckets.values())
    median = sizes[len(sizes) // 2]
    cap = max(5, int(median * 1.5))

    balanced, before = [], {str(k): len(v) for k, v in buckets.items()}
    for key, bucket in buckets.items():
        random.shuffle(bucket)
        balanced.extend(bucket[:cap])
    random.shuffle(balanced)
    after = Counter((r["domain"],
                     "regression_value" if r["prediction_class"].startswith("$")
                     else r["prediction_class"]) for r in balanced)
    return balanced, {"before": before, "after": {str(k): v for k, v in after.items()},
                      "cap_per_bucket": cap}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build + optimize the Explainer fine-tuning dataset")
    parser.add_argument("--n-synthetic", type=int, default=800,
                        help="Number of synthetic scenarios to generate")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Write explanations with local Mistral via Ollama "
                             "(higher quality/diversity; slower). Falls back to "
                             "templates automatically if the server is down.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    use_ollama = args.use_ollama and ollama_available()
    if args.use_ollama and not use_ollama:
        print("! Ollama not reachable — falling back to template writers.")
    print(f"Writer mode: {'Ollama (' + OLLAMA_MODEL + ')' if use_ollama else 'deterministic templates'}")

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Generating {args.n_synthetic} synthetic scenarios "
          f"across {len(DOMAINS)} domains...")
    rows = []
    domains = list(DOMAINS.keys())
    for i in range(args.n_synthetic):
        scenario = generate_scenario(domains[i % len(domains)])
        good, bad = write_explanations(scenario, use_ollama)
        scenario["good"], scenario["bad"] = good, bad
        rows.append(scenario)
        if use_ollama and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_synthetic} written...")

    real = harvest_real_examples()
    print(f"[2/5] Harvested {len(real)} real examples from past pipeline runs")
    for r in real:
        r.setdefault("bad", template_bad_explanation(r) if "- " in r["shap_input"] else
                     "The model considered many factors and made this prediction.")
    rows.extend(real)
    n_raw = len(rows)

    # ── Optimize ──────────────────────────────────────────────────────────
    print(f"[3/5] Optimizing dataset ({n_raw} raw examples)...")
    rows, n_dupes = dedupe(rows)
    rows, filter_reasons = quality_filter(rows)
    rows, balance_report = balance(rows)
    print(f"  removed {n_dupes} duplicates, "
          f"filtered {sum(filter_reasons.values())} low-quality "
          f"({filter_reasons}), kept {len(rows)} after balancing")

    # ── Split ─────────────────────────────────────────────────────────────
    print("[4/5] Splitting train/val/test...")
    random.shuffle(rows)
    n_test = int(len(rows) * args.test_frac)
    n_val = int(len(rows) * args.val_frac)
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]

    sft = lambda subset: [{"input": r["shap_input"], "output": r["good"],
                           "domain": r["domain"]} for r in subset]
    prefs = lambda subset: [{"prompt": r["shap_input"], "chosen": r["good"],
                             "rejected": r["bad"]} for r in subset]

    save_jsonl(sft(train), TRAIN_FILE)
    save_jsonl(sft(val), VAL_FILE)
    save_jsonl(sft(test), TEST_FILE)
    save_jsonl(prefs(train), PREF_TRAIN_FILE)
    save_jsonl(prefs(val), PREF_VAL_FILE)

    # ── Report ────────────────────────────────────────────────────────────
    print("[5/5] Writing dataset report...")
    domain_counts = Counter(r["domain"] for r in rows)
    avg_in = sum(len(r["shap_input"]) for r in rows) / max(len(rows), 1)
    avg_out = sum(len(r["good"]) for r in rows) / max(len(rows), 1)
    report = f"""# Explainer Fine-Tuning Dataset Report

| Stage | Count |
|---|---|
| Raw examples (synthetic + real) | {n_raw} |
| After exact/near-duplicate removal | {n_raw - n_dupes} |
| After quality filtering | {n_raw - n_dupes - sum(filter_reasons.values())} |
| After class balancing (final) | {len(rows)} |

**Filter breakdown:** {json.dumps(filter_reasons)}
**Balance cap per (domain, class) bucket:** {balance_report['cap_per_bucket']}

## Splits
| Split | Examples |
|---|---|
| train | {len(train)} |
| val | {len(val)} |
| test | {len(test)} |
| preference pairs (train) | {len(train)} |
| preference pairs (val) | {len(val)} |

## Domain coverage (final)
| Domain | Examples |
|---|---|
""" + "\n".join(f"| {d} | {c} |" for d, c in domain_counts.most_common()) + f"""

## Text statistics
- Average input length: {avg_in:.0f} chars
- Average output length: {avg_out:.0f} chars
- Writer mode: {'Ollama ' + OLLAMA_MODEL if use_ollama else 'deterministic templates'}
- Real examples harvested from past runs: {len(real)}
"""
    report_path = os.path.join(DATA_DIR, "dataset_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nDone. Report: {report_path}")
    print(f"SFT data: {TRAIN_FILE} (+val/test)")
    print(f"Preference data: {PREF_TRAIN_FILE} (+val)")


if __name__ == "__main__":
    main()
