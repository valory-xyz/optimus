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

"""Sizing invariants for the Optimism agent-EOA fund_requirements.

OPE-1940: an agent EOA sitting above its native threshold but below what an
x402 USDC top-up swap costs can never afford the swap and never reports a
standard deficit, so it stalls silently. These assertions keep that arithmetic
machine-checked instead of leaving it to a reviewer.
"""

import json
import re
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
SERVICE_YAML = REPO_ROOT / "packages/valory/services/optimus/service.yaml"
AGENT_YAML = REPO_ROOT / "packages/valory/agents/optimus/aea-config.yaml"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# The x402 top-up swap buys 0.25 USDC, per the x402_payment_requirements
# override both service.yaml files carry.
X402_TOPUP_USDC = 0.25

# Conservative low ETH price, in USD. The ETH value of a fixed-USDC swap rises
# as the ETH price falls, so the sizing has to hold at a price well below
# today's rather than at the snapshot the ticket was written against (~$2734).
# Raise this only with evidence; lowering it tightens the invariant.
CONSERVATIVE_ETH_PRICE_USD = 1500

WEI_PER_ETH = 10**18
SINGLE_SWAP_WEI = int(X402_TOPUP_USDC / CONSERVATIVE_ETH_PRICE_USD * WEI_PER_ETH)


def _parse_fund_requirements(path: Path) -> Dict:
    """Extract the fund_requirements default from an Open Autonomy yaml.

    Both files wrap the value as ``${[NAME:]dict:{...}}``, so the JSON payload
    runs from the first brace after ``dict:`` to the closing brace of the
    substitution.

    :param path: the yaml file to read.
    :return: the parsed fund_requirements mapping.
    """
    for line in path.read_text().splitlines():
        if "fund_requirements:" not in line:
            continue
        match = re.search(r"dict:(\{.*\})\}\s*$", line.strip())
        assert match is not None, f"unrecognised fund_requirements syntax in {path}"
        return json.loads(match.group(1))
    raise AssertionError(f"no fund_requirements entry in {path}")


@pytest.mark.parametrize("path", [SERVICE_YAML, AGENT_YAML])
def test_optimism_agent_eoa_can_afford_a_swap(path: Path) -> None:
    """The agent EOA threshold and topup must both clear one swap's cost.

    A threshold below one swap's cost is precisely how an agent reaches "above
    threshold, cannot afford a swap, reports nothing".

    :param path: the configuration file under test.
    """
    agent_native = _parse_fund_requirements(path)["optimism"]["agent"][ZERO_ADDRESS]
    assert agent_native["threshold"] >= SINGLE_SWAP_WEI
    assert agent_native["topup"] > agent_native["threshold"]


@pytest.mark.parametrize("path", [SERVICE_YAML, AGENT_YAML])
def test_optimism_agent_eoa_topup_covers_multiple_swaps(path: Path) -> None:
    """The topup must fund more than one swap, or the prompt recurs every cycle.

    :param path: the configuration file under test.
    """
    agent_native = _parse_fund_requirements(path)["optimism"]["agent"][ZERO_ADDRESS]
    assert agent_native["topup"] >= 2 * SINGLE_SWAP_WEI


def test_service_and_agent_fund_requirements_agree() -> None:
    """The service override and the agent default must not drift apart."""
    assert _parse_fund_requirements(SERVICE_YAML) == _parse_fund_requirements(
        AGENT_YAML
    )
