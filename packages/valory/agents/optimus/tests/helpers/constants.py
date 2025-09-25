# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""Defines constants used in the Optimus docker image."""

# the network is configured with these accounts
ACCOUNTS = [
    (
        "0xFFcf8FDEE72ac11b5c542428B35EEF5769C409f0",
        "0x6cbed15c793ce57650b9877cf6fa156fbef513c4e6134f022a85b1ffdd59b2a1",
    ),
    (
        "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1",
        "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
    ),
    (
        "0x22d491Bde2303f2f43325b2108D26f1eAbA1e32b",
        "0x6370fd033278c143179d81c5526140625662b8daa446c22ee2d73db3707e620c",
    ),
    (
        "0xE11BA2b4D45Eaed5996Cd0823791E0C93114882d",
        "0x646f1ce2fdad0e6deeeb5c7e8e5543bdde65e86029e2fd9fc169899c440a7913",
    ),
]

# NOTE: these addresses are deterministic for Optimism
SAFE_CONTRACT_ADDRESS = "0x1234567890123456789012345678901234567890"
MULTISEND_CONTRACT_ADDRESS = "0xbE5b0013D2712DC4faF07726041C27ecFdBC35AD"
BALANCER_VAULT_ADDRESS = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"

# Optimism token addresses
OPTIMISM_TOKENS = {
    "ETH": "0x0000000000000000000000000000000000000000",
    "USDC": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
}

# Chain ID for Optimism
OPTIMISM_CHAIN_ID = 10
