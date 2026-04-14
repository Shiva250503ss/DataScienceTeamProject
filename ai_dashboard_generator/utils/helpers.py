from __future__ import annotations

import re
from typing import List, Optional


BUSINESS_NUMERIC_HINTS = [
    "sales", "revenue", "amount", "price", "profit", "cost", "qty", "quantity",
    "units", "expense", "spend", "score", "rating", "count", "total", "value"
]

DATE_HINTS = ["date", "time", "day", "month", "year", "created", "updated"]
CATEGORY_HINTS = ["category", "type", "segment", "region", "store", "city", "state"]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def pick_first_matching_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    lowered = {col: col.lower() for col in columns}
    for keyword in keywords:
        for original, low in lowered.items():
            if keyword in low:
                return original
    return None


def safe_list_remove(items: List[str], values_to_remove: List[str]) -> List[str]:
    to_remove = set(values_to_remove)
    return [item for item in items if item not in to_remove]
