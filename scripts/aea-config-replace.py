#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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


"""Updates fetched agent with correct config"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def main() -> None:
    """Main"""
    load_dotenv()

    # Check if data directory exists, create if it doesn't
    data_dir = Path("data")
    if data_dir.exists():
        print(f"Using existing data directory: {data_dir.absolute()}")
    else:
        data_dir.mkdir(exist_ok=True)
        print(f"Created new data directory: {data_dir.absolute()}")

    with open(Path("optimus", "aea-config.yaml"), "r", encoding="utf-8") as file:
        config = list(yaml.safe_load_all(file))

        # Find the ledger connection config (should be at index 4 based on the YAML structure)
        ledger_config_index = None
        for i, doc in enumerate(config):
            if (isinstance(doc, dict) and 
                doc.get("public_id") == "valory/ledger:0.19.0" and 
                doc.get("type") == "connection"):
                ledger_config_index = i
                break
        
        if ledger_config_index is not None and os.getenv("MODE_LEDGER_RPC"):
            config[ledger_config_index]["config"]["ledger_apis"]["mode"][
                "address"
            ] = f"${{str:{os.getenv('MODE_LEDGER_RPC')}}}"

        # Find the optimus_abci skill config (should be at index 7 based on the YAML structure)
        skill_config_index = None
        for i, doc in enumerate(config):
            if (isinstance(doc, dict) and 
                doc.get("public_id") == "valory/optimus_abci:0.1.0" and 
                doc.get("type") == "skill"):
                skill_config_index = i
                break

        # Find the kv_store connection config
        kv_store_config_index = None
        for i, doc in enumerate(config):
            if (isinstance(doc, dict) and 
                doc.get("public_id") == "dvilela/kv_store:0.1.0" and 
                doc.get("type") == "connection"):
                kv_store_config_index = i
                break
        
        if kv_store_config_index is not None:
            # Update store_path to point to the local data directory
            config[kv_store_config_index]["config"]["store_path"] = f"${{str:{data_dir.absolute()}}}"
            print(f"Updated store_path to: {data_dir.absolute()}")

        if skill_config_index is not None:
            # Update store_path in skill config as well
            config[skill_config_index]["models"]["params"]["args"]["store_path"] = f"${{str:{data_dir.absolute()}}}"
            
            # Update log_dir in benchmark_tool to use data directory
            config[skill_config_index]["models"]["benchmark_tool"]["args"]["log_dir"] = f"${{str:{data_dir.absolute()}}}"

            # Params
            if os.getenv('ALL_PARTICIPANTS'):
                config[skill_config_index]["models"]["params"]["args"]["setup"][
                    "all_participants"
                ] = f"${{list:{os.getenv('ALL_PARTICIPANTS')}}}"

            if os.getenv('SAFE_CONTRACT_ADDRESSES'):
                config[skill_config_index]["models"]["params"]["args"][
                    "safe_contract_addresses"
                ] = f"${{str:{os.getenv('SAFE_CONTRACT_ADDRESSES')}}}"

            if os.getenv('TENDERLY_ACCESS_KEY'):
                config[skill_config_index]["models"]["params"]["args"][
                    "tenderly_access_key"
                ] = f"${{str:{os.getenv('TENDERLY_ACCESS_KEY')}}}"

            if os.getenv('TENDERLY_ACCOUNT_SLUG'):
                config[skill_config_index]["models"]["params"]["args"][
                    "tenderly_account_slug"
                ] = f"${{str:{os.getenv('TENDERLY_ACCOUNT_SLUG')}}}"

            if os.getenv('TENDERLY_PROJECT_SLUG'):
                config[skill_config_index]["models"]["params"]["args"][
                    "tenderly_project_slug"
                ] = f"${{str:{os.getenv('TENDERLY_PROJECT_SLUG')}}}"

            if os.getenv('COINGECKO_API_KEY'):
                config[skill_config_index]["models"]["coingecko"]["args"][
                    "api_key"
                ] = f"${{str:{os.getenv('COINGECKO_API_KEY')}}}"

            if os.getenv('TARGET_INVESTMENT_CHAINS'):
                config[skill_config_index]["models"]["params"]["args"][
                    "target_investment_chains"
                ] = f"${{list:{os.getenv('TARGET_INVESTMENT_CHAINS')}}}"

    with open(Path("optimus", "aea-config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump_all(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
