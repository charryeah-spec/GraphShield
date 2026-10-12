#!/usr/bin/env python3
"""Create a machine-readable leakage report before any model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.leakage_audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to data.npz")
    parser.add_argument(
        "--output", default="artifacts/data_audit.json", help="JSON output path"
    )
    parser.add_argument(
        "--skip-permutation-check", action="store_true", help="skip edge-row shuffle audit"
    )
    args = parser.parse_args()

    report = audit_dataset(args.data, not args.skip_permutation_check)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["risk_level"] == "critical":
        print("\nCRITICAL: unsafe temporal proxies must remain disabled.", file=sys.stderr)


if __name__ == "__main__":
    main()
