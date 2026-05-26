"""Select a prioritized balanced list from auto-label metadata."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "autolabel_uploads" / "metadata.csv"
SELECTED = ROOT / "dataset" / "autolabel_uploads" / "selected_balanced.txt"

ISSUE_ORDER = ["led_magenta", "none", "led_purple", "mixed"]
QUOTAS = {"led_magenta": 20, "none": 20, "led_purple": 15, "mixed": 15}


def confidence(row: dict[str, str]) -> float:
    return float(row.get("avg_confidence") or 0)


def main() -> None:
    rows = list(csv.DictReader(META.open()))
    by_issue = {issue: [] for issue in ISSUE_ORDER}
    for row in rows:
        issue = row["issue"]
        if issue in by_issue:
            by_issue[issue].append(row)
        else:
            by_issue["mixed"].append(row)

    for issue_rows in by_issue.values():
        issue_rows.sort(key=lambda row: -confidence(row))

    selected: list[dict[str, str]] = []
    selected_names: set[str] = set()
    selected_counts = {issue: 0 for issue in ISSUE_ORDER}

    for issue in ISSUE_ORDER:
        for row in by_issue[issue][: QUOTAS[issue]]:
            selected.append(row)
            selected_names.add(row["filename"])
            selected_counts[issue] += 1

    # If quotas cannot be met (for example no mixed uploads), keep all 70 names
    # prioritized by confidence, preferring additional magenta if available.
    remaining = [row for row in rows if row["filename"] not in selected_names]
    priority_rank = {"led_magenta": 0, "none": 1, "led_purple": 2}
    remaining.sort(key=lambda row: (priority_rank.get(row["issue"], 3), -confidence(row)))
    for row in remaining:
        selected.append(row)
        issue = row["issue"] if row["issue"] in selected_counts else "mixed"
        selected_counts[issue] += 1

    SELECTED.write_text("\n".join(row["filename"] for row in selected) + ("\n" if selected else ""))
    print(f"Selecionados {len(selected)} pra balanceado: {SELECTED}")
    for issue in ISSUE_ORDER:
        print(f"  {issue}: {selected_counts[issue]}")


if __name__ == "__main__":
    main()
