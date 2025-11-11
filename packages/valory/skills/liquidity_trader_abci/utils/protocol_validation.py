"""Protocol validation utilities for the liquidity trader skill."""

from typing import List, Optional


def validate_and_fix_protocols(
    selected_protocols: List[str],
    target_investment_chains: List[str],
    available_strategies: dict,
    previous_protocols: Optional[List[str]] = None,
) -> List[str]:
    """Validate selected protocols and fix invalid ones without resetting to defaults.
    
    Args:
        selected_protocols: List of protocol names to validate
        target_investment_chains: List of chains to check availability on
        available_strategies: Dictionary mapping chains to available strategies
        previous_protocols: Optional list of previously selected protocols to preserve (grandfathering)
    
    Returns:
        List of validated protocol names
    """
    # Valid protocol names
    # Note: uniswapV3 is kept here for grandfathering existing users, but it's removed from
    # available_strategies and chat UI prompts. New users won't get it, but existing users
    # who already have it selected will keep it when changing strategies.
    VALID_PROTOCOLS = {
        "balancerPool": "balancer_pools_search",
        "uniswapV3": "uniswap_pools_search",  # Deprecated for new users, kept for grandfathering
        "velodrome": "velodrome_pools_search",
        "sturdy": "asset_lending",
    }

    # Normalize previous_protocols to a set for easy lookup
    previous_protocols_set = set(previous_protocols) if previous_protocols else set()

    # Check if any protocol is invalid or not available on selected chains
    invalid_protocols = []
    valid_protocols = []
    chain_incompatible_protocols = []

    for protocol in selected_protocols:
        if protocol not in VALID_PROTOCOLS:
            invalid_protocols.append(protocol)
            continue

        # Check if protocol is available on ANY of the target chains
        protocol_available = False
        for chain in target_investment_chains:
            if chain in available_strategies:
                chain_strategies = available_strategies[chain]
                strategy_name = VALID_PROTOCOLS[protocol]
                if strategy_name in chain_strategies:
                    protocol_available = True
                    break

        if protocol_available:
            valid_protocols.append(protocol)
        else:
            # If protocol is not available but was previously selected, preserve it (grandfathering)
            if protocol in previous_protocols_set:
                valid_protocols.append(protocol)
            else:
                chain_incompatible_protocols.append(protocol)

    # If any invalid or incompatible protocols found, get fallback protocols
    if invalid_protocols or chain_incompatible_protocols:
        # Get default protocols for all target chains as fallback
        default_protocols = []
        for chain in target_investment_chains:
            if chain in available_strategies:
                chain_strategies = available_strategies[chain]
                # Convert strategies to protocol names
                for strategy in chain_strategies:
                    for protocol, strategy_name in VALID_PROTOCOLS.items():
                        if strategy == strategy_name:
                            if protocol not in default_protocols:
                                default_protocols.append(protocol)

        # If we have valid protocols, keep them - don't add defaults unless ALL are invalid
        if valid_protocols:
            return valid_protocols
        else:
            # Only if ALL protocols are invalid, use defaults
            return default_protocols

    return valid_protocols
