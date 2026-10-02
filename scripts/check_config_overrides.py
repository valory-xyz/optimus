#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""Check every agent override names a key the overridden component declares.

The framework refuses an override whose path the component does not declare, and
the only thing that exercises that is the agent image build, which runs at
release time. A composed skill redeclares the models of the skills it composes,
so a parameter added to the underlying skill and to the agent, but not to the
composed one, passes every other check and then fails the release.

Only agent configurations are checked. Those are the overrides the image build
applies; service overrides are validated elsewhere against a different pattern,
and including them reports differences the build accepts.

Takes no arguments. Run from the repository root.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml


PACKAGES = Path("packages")
CONFIG_FILE = {
    "skill": "skill.yaml",
    "connection": "connection.yaml",
    "contract": "contract.yaml",
    "protocol": "protocol.yaml",
}
# Keys naming the override's target rather than forming part of its payload.
TARGET_KEYS = ("public_id", "type")


def load_docs(path: Path) -> List[Dict[str, Any]]:
    """Return every mapping document in a YAML file.

    :param path: the file to read.
    :return: the documents it holds.
    """
    return [d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)]


def component_config(public_id: str, kind: str) -> Dict[str, Any]:
    """Return the configuration a component declares for itself.

    :param public_id: the overridden component's public id.
    :param kind: its component type.
    :return: its own configuration, or an empty mapping when the component is
        not vendored here, leaving nothing local to check against.
    """
    author, rest = public_id.split("/", 1)
    name = rest.split(":", 1)[0]
    path = PACKAGES / author / f"{kind}s" / name / CONFIG_FILE.get(kind, "")
    if not path.is_file():
        return {}
    docs = load_docs(path)
    return docs[0] if docs else {}


def declared_paths(
    node: Any, prefix: Tuple[str, ...] = ()
) -> Iterator[Tuple[str, ...]]:
    """Yield every path a configuration declares, intermediate ones included.

    :param node: the configuration to walk.
    :param prefix: the path accumulated so far.
    :yield: each declared path.
    """
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield (*prefix, str(key))
            yield from declared_paths(value, (*prefix, str(key)))


def overridden_paths(
    node: Any, prefix: Tuple[str, ...] = ()
) -> Iterator[Tuple[str, ...]]:
    """Yield the path of every value an override sets.

    :param node: the override payload.
    :param prefix: the path accumulated so far.
    :yield: each overridden leaf path.
    """
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from overridden_paths(value, (*prefix, str(key)))
    else:
        yield prefix


def check(path: Path) -> List[str]:
    """Check one agent configuration.

    :param path: the aea-config.yaml to check.
    :return: one message per override the target does not declare.
    """
    problems: List[str] = []
    for doc in load_docs(path)[1:]:
        public_id, kind = doc.get("public_id"), doc.get("type")
        if not public_id or not kind:
            continue
        config = component_config(str(public_id), str(kind))
        if not config:
            continue
        declared = set(declared_paths(config))
        payload = {k: v for k, v in doc.items() if k not in TARGET_KEYS}
        for overridden in overridden_paths(payload):
            if overridden and overridden not in declared:
                problems.append(
                    f"{path}\n  {public_id} does not declare "
                    f"`{'.'.join(overridden)}`, so the agent image build "
                    "refuses the override"
                )
    return problems


def main() -> int:
    """Check every agent configuration in the repository.

    :return: the process exit status.
    """
    targets = sorted(PACKAGES.glob("*/agents/*/aea-config.yaml"))
    if not targets:
        print("no agent configurations found", file=sys.stderr)
        return 1

    problems: List[str] = []
    for target in targets:
        problems.extend(check(target))

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"\n{len(problems)} override(s) the target does not declare. Add them "
            "to the overridden component, or stop overriding them.",
            file=sys.stderr,
        )
        return 1
    print(f"checked {len(targets)} agent configuration(s); every override is declared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
