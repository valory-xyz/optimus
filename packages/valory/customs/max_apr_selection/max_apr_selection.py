# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This module contains a strategy that selects the best yielding strategy based on APR"""

from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Union,
)

REQUIRED_FIELDS = ("trading_opportunities",)


def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    return {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}

def select_best_opportunity(
    trading_opportunities: List[Any]
) -> Dict[str, Union[bool, str]]:
    """Select the best trading opportunity"""
    selected_opportunity = None
    highest_apr = -float("inf")
    for trading_opportunity in trading_opportunities:
        if trading_opportunity is not None:
            apr = trading_opportunity.get("apr", 0)
            if apr > highest_apr:
                highest_apr = apr
                selected_opportunity = trading_opportunity
    
    return selected_opportunity

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    return select_best_opportunity(**kwargs)