"""Build a tiny 1-sample / 3-QA subset of locomo10.json for smoke testing.

Run locally (no GPU needed):
    python scripts/make_smoke_subset.py

Writes: data/locomo10_smoke.json
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "locomo10.json"
DST = REPO_ROOT / "data" / "locomo10_smoke.json"

SAMPLE_ID = "conv-30"   # fewest QAs per locomo_sample_summary.csv
NUM_QAS = None

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

picked = next((s for s in data if s.get("sample_id") == SAMPLE_ID), None)
if picked is None:
    raise SystemExit(f"sample_id {SAMPLE_ID!r} not found in {SRC}")

picked = dict(picked)
picked["qa"] = picked["qa"][:NUM_QAS]

with open(DST, "w", encoding="utf-8") as f:
    json.dump([picked], f, ensure_ascii=False, indent=2)

print(f"Wrote {DST}")
print(f"sample_id={picked['sample_id']}  qas={len(picked['qa'])}  sessions_in_conv={sum(1 for k in picked.get('conversation', {}) if k.startswith('session_') and not k.endswith('_date_time'))}")
