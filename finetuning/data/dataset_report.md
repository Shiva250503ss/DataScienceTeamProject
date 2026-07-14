# Explainer Fine-Tuning Dataset Report

| Stage | Count |
|---|---|
| Raw examples (synthetic + real) | 800 |
| After exact/near-duplicate removal | 800 |
| After quality filtering | 800 |
| After class balancing (final) | 756 |

**Filter breakdown:** {}
**Balance cap per (domain, class) bucket:** 78

## Splits
| Split | Examples |
|---|---|
| train | 606 |
| val | 75 |
| test | 75 |
| preference pairs (train) | 606 |
| preference pairs (val) | 75 |

## Domain coverage (final)
| Domain | Examples |
|---|---|
| employee_attrition | 100 |
| medical_readmission | 100 |
| marketing_conversion | 100 |
| fraud_detection | 100 |
| customer_churn | 100 |
| credit_risk | 100 |
| sales_forecast | 78 |
| house_price | 78 |

## Text statistics
- Average input length: 449 chars
- Average output length: 366 chars
- Writer mode: deterministic templates
- Real examples harvested from past runs: 0
