from __future__ import annotations

import math


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            if abs(value) >= 1_000_000:
                return f"{value:,.2f}"
            if float(value).is_integer():
                return f"{int(value):,}"
            return f"{value:,.2f}"
    return str(value)
