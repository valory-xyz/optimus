import math
import logging
from typing import Any, Dict, List, Union

# Constants
REQUIRED_FIELDS = ("trading_opportunities","current_pool")
SHARPE_RATIO_THRESHOLD = 1
DEPTH_SCORE_THRESHOLD = 50
IL_RISK_SCORE_THRESHOLD = -0.05

# Weights for each metric
SHARPE_RATIO_WEIGHT = 0.4
DEPTH_SCORE_WEIGHT = 0.3
IL_RISK_SCORE_WEIGHT = 0.3

# Configure logging
logging.basicConfig(level=logging.INFO)

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

def calculate_composite_score(pool, max_values):
    """Calculate the composite score for a given pool."""
    sharpe_ratio = pool.get("sharpe_ratio", 0)
    depth_score = pool.get("depth_score", 0)
    il_risk_score = pool.get("il_risk_score", float('inf'))

    if not isinstance(sharpe_ratio, (int, float)):
        sharpe_ratio = 0
    if not isinstance(depth_score, (int, float)):
        depth_score = 0
    if not isinstance(il_risk_score, (int, float)):
        il_risk_score = float('inf')

    # Normalize metrics
    normalized_sharpe_ratio = sharpe_ratio / max_values['sharpe_ratio']
    normalized_depth_score = depth_score / max_values['depth_score']
    normalized_il_risk_score = (max_values['il_risk_score'] - il_risk_score) / max_values['il_risk_score']

    # Calculate composite score
    return (
        SHARPE_RATIO_WEIGHT * normalized_sharpe_ratio +
        DEPTH_SCORE_WEIGHT * normalized_depth_score +
        IL_RISK_SCORE_WEIGHT * normalized_il_risk_score
    )

def get_max_values(pools):
    """Get maximum values for normalization."""
    return {
        'sharpe_ratio': max((pool.get("sharpe_ratio", 0) for pool in pools)),
        'depth_score': max((pool.get("depth_score", 0) for pool in pools)),
        'il_risk_score': max((pool.get("il_risk_score", 0) for pool in pools))
    }

def apply_risk_thresholds_and_select_optimal_strategy(trading_opportunities, current_pool=None, improvement_threshold=0.1):
    """Apply risk thresholds and select the optimal strategy based on combined metrics."""
    
    # Filter opportunities based on risk thresholds
    filtered_opportunities = []
    for opportunity in trading_opportunities:
        sharpe_ratio = opportunity.get("sharpe_ratio", 0)
        depth_score = opportunity.get("depth_score", 0)
        il_risk_score = opportunity.get("il_risk_score", float('inf'))

        print(f"Evaluating opportunity: {opportunity}")
        if not isinstance(sharpe_ratio, (int, float)) or not isinstance(depth_score, (int, float)) or not isinstance(il_risk_score, (int, float)):
            print(f"Invalid values for risk metrics")
            continue

        if sharpe_ratio <= SHARPE_RATIO_THRESHOLD:
            print(f"Opportunity does not meet the {SHARPE_RATIO_THRESHOLD=}")
            continue

        if depth_score <= DEPTH_SCORE_THRESHOLD:
            print(f"Opportunity does not meet the {DEPTH_SCORE_THRESHOLD=}")
            continue

        if il_risk_score <= IL_RISK_SCORE_THRESHOLD:
            print(f"Opportunity does not meet the {IL_RISK_SCORE_THRESHOLD=}")
            continue

        print("Opportunity meets all risk thresholds")
        filtered_opportunities.append(opportunity)

    if not filtered_opportunities:
        logging.error("No opportunities meet the risk thresholds.")
        return {}

    if not filtered_opportunities:
        logging.error("No opportunities meet the risk thresholds.")
        return {}

    # Calculate max values for normalization
    max_values = get_max_values(filtered_opportunities)

    # Calculate composite scores for filtered opportunities
    for opportunity in filtered_opportunities:
        opportunity["composite_score"] = calculate_composite_score(opportunity, max_values)

    if current_pool:
        # Calculate composite score for the current pool
        current_composite_score = calculate_composite_score(current_pool, max_values)
        # Compare each opportunity with the current pool
        better_opportunities = [
            opportunity for opportunity in filtered_opportunities
            if opportunity["composite_score"] > current_composite_score * (1 + improvement_threshold)
        ]

        if better_opportunities:
            # Select the best opportunity
            optimal_opportunity = max(better_opportunities, key=lambda x: x["composite_score"])
        else:
            logging.error("No opportunities significantly better than the current pool.")
            return {}
    else:
        # Select the optimal opportunity (e.g., top opportunity by highest composite score)
        optimal_opportunity = max(filtered_opportunities, key=lambda x: x["composite_score"], default=None)
    
    return optimal_opportunity

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    optimal_strategy = apply_risk_thresholds_and_select_optimal_strategy(**kwargs)
    return optimal_strategy