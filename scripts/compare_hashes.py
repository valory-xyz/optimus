#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Compare package hashes between source-of-truth repos and target repo."""

import json
from pathlib import Path

# Source of truth repos (dev packages become third_party in downstream)
SOURCE_REPOS = [
    Path("/Users/dhairya/Desktop/Work/Valory/Github/open-aea/packages/packages.json"),
    Path("/Users/dhairya/Desktop/Work/Valory/Github/genai/packages/packages.json"),
    Path(
        "/Users/dhairya/Desktop/Work/Valory/Github/funds-manager/packages/packages.json"
    ),
    Path("/Users/dhairya/Desktop/Work/Valory/Github/kv-store/packages/packages.json"),
    Path(
        "/Users/dhairya/Desktop/Work/Valory/Github/open-autonomy/packages/packages.json"
    ),
]

# Target: this repo's packages.json
TARGET_PACKAGES_JSON = Path("packages/packages.json")


def main() -> None:
    """Compare hashes."""
    # Merge all source packages
    source_all = {}
    for source_path in SOURCE_REPOS:
        if not source_path.exists():
            print(f"WARNING: Source not found: {source_path}")
            continue
        with open(source_path, encoding="utf-8") as f:
            source = json.load(f)
        source_dev = source.get("dev", {})
        source_third = source.get("third_party", {})
        source_all.update(source_third)
        source_all.update(source_dev)

    with open(TARGET_PACKAGES_JSON, encoding="utf-8") as f:
        target = json.load(f)

    target_third = target.get("third_party", {})

    mismatches = []
    missing = []
    for pkg, target_hash in target_third.items():
        if pkg in source_all:
            source_hash = source_all[pkg]
            if target_hash != source_hash:
                mismatches.append((pkg, target_hash, source_hash))
        else:
            missing.append(pkg)

    if missing:
        print(f"Packages not found in any source repo ({len(missing)}):")
        for pkg in missing:
            print(f"  {pkg}")
        print()

    if not mismatches:
        print("All hashes match!")
        return

    print(f"Found {len(mismatches)} mismatched hashes:\n")
    for pkg, _old, new in mismatches:
        print(f'  "{pkg}": "{new}",')
    print(f"\nReplace these in {TARGET_PACKAGES_JSON}")


if __name__ == "__main__":
    main()
