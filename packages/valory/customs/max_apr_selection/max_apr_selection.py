import math
from typing import Any, Dict, List, Union

# Constants
REQUIRED_FIELDS = ("trading_opportunities", "current_positions", "max_pools")
SHARPE_RATIO_THRESHOLD = 1
DEPTH_SCORE_THRESHOLD = 50
IL_RISK_SCORE_THRESHOLD = -0.2

# Weights for each metric
SHARPE_RATIO_WEIGHT = 0.4
DEPTH_SCORE_WEIGHT = 0.3
IL_RISK_SCORE_WEIGHT = 0.3
MIN_COMPOSITE_SCORE_RATIO = 0.5

logs = []
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
    sharpe_ratio = pool.get("sharpe_ratio", math.nan)
    depth_score = pool.get("depth_score", math.nan)
    il_risk_score = pool.get("il_risk_score", math.nan)

    if math.isnan(sharpe_ratio) or math.isnan(depth_score) or math.isnan(il_risk_score):
        return 0

    # Normalize metrics
    normalized_sharpe_ratio = sharpe_ratio / max_values["sharpe_ratio"]
    normalized_depth_score = depth_score / max_values["depth_score"]
    normalized_il_risk_score = (abs(il_risk_score)) / abs(max_values["il_risk_score"])

    # Calculate composite score
    return (
        SHARPE_RATIO_WEIGHT * normalized_sharpe_ratio
        + DEPTH_SCORE_WEIGHT * normalized_depth_score
        + IL_RISK_SCORE_WEIGHT * normalized_il_risk_score
    )


def get_max_values(pools):
    """Get maximum values for normalization."""
    return {
        "sharpe_ratio": max((pool.get("sharpe_ratio", 0) for pool in pools)),
        "depth_score": max((pool.get("depth_score", 0) for pool in pools)),
        "il_risk_score": max((pool.get("il_risk_score", 0) for pool in pools)),
    }


def calculate_relative_percentages(percentages):
    """
    Calculate the relative percentages of a list of percentages.
    This function takes a list of percentages and calculates the relative
    percentage of each element with respect to the running total of the
    percentages encountered so far.
    Args:
        percentages (list of float): A list of percentages.
    Returns:
        list of float: A list of relative percentages.
    """
    total_percentage = sum(percentages)
    dynamic_percentages = []

    if total_percentage == 0:
        logs.append("ERROR: Total percentage cannot be zero.")
        return []

    for percentage in percentages:
        dynamic_percentage = percentage / total_percentage
        dynamic_percentages.append(dynamic_percentage)
        total_percentage -= percentage

    return dynamic_percentages


def apply_risk_thresholds_and_select_optimal_strategy(
    trading_opportunities,
    current_positions=None,
    improvement_threshold=0.1,
    max_pools=1,
):
    """Apply risk thresholds and select the optimal strategy based on combined metrics."""

    # Filter opportunities based on risk thresholds
    filtered_opportunities = []
    for opportunity in trading_opportunities:
        sharpe_ratio = opportunity.get("sharpe_ratio", 0)
        depth_score = opportunity.get("depth_score", 0)
        il_risk_score = opportunity.get("il_risk_score", float("inf"))

        logs.append(f"Evaluating opportunity: {opportunity}")
        if (
            not isinstance(sharpe_ratio, (int, float))
            or not isinstance(depth_score, (int, float))
            or not isinstance(il_risk_score, (int, float))
        ):
            logs.append("WARNING: Invalid values for risk metrics")
            continue

        if sharpe_ratio <= SHARPE_RATIO_THRESHOLD:
            logs.append(f"Opportunity does not meet the {SHARPE_RATIO_THRESHOLD=}")
            continue

        if depth_score <= DEPTH_SCORE_THRESHOLD:
            logs.append(f"Opportunity does not meet the {DEPTH_SCORE_THRESHOLD=}")
            continue

        if il_risk_score <= IL_RISK_SCORE_THRESHOLD:
            logs.append(f"Opportunity does not meet the {IL_RISK_SCORE_THRESHOLD=}")
            continue

        logs.append("Opportunity meets all risk thresholds")
        filtered_opportunities.append(opportunity)

    if not filtered_opportunities:
        logs.append("No opportunities meet the risk thresholds.")
        return {}

    # Calculate max values for normalization
    max_values = get_max_values(filtered_opportunities)

    # Calculate composite scores for filtered opportunities
    for opportunity in filtered_opportunities:
        opportunity["composite_score"] = calculate_composite_score(
            opportunity, max_values
        )

    position_to_exit = {}
    optimal_opportunities = []

    if current_positions:
        # Calculate composite score for the current pool
        current_composite_scores = [
            calculate_composite_score(pool, max_values) for pool in current_positions
        ]

        # Identify the least performing current pool
        least_performing_index = current_composite_scores.index(
            min(current_composite_scores)
        )
        least_performing_score = current_composite_scores[least_performing_index]
        position_to_exit = current_positions[least_performing_index]

        # Compare each opportunity with the least performing current pool
        better_opportunities = [
            opportunity
            for opportunity in filtered_opportunities
            if opportunity["composite_score"]
            > least_performing_score * (1 + improvement_threshold)
        ]

        if better_opportunities:
            # Sort and select the top opportunity
            better_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
            optimal_opportunities = [better_opportunities[0]]
            optimal_opportunities[0]["relative_funds_percentage"] = 1.0
            logs.append(
                f"Top opportunity found with composite score: {optimal_opportunities[0]['composite_score']}"
            )
        else:
            logs.append.warning(
                f"No opportunities significantly better than the least performing current opportunity with composite score: {least_performing_score}"
            )
            return {"optimal_strategies": [], "position_to_exit": {}}
    else:
        # Sort opportunities based on composite score in descending order
        filtered_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        top_composite_score = filtered_opportunities[0]["composite_score"]

        # Select opportunities that meet the minimum composite score ratio
        optimal_opportunities = [
            opp
            for opp in filtered_opportunities[:max_pools]
            if opp["composite_score"] >= MIN_COMPOSITE_SCORE_RATIO * top_composite_score
        ]

        if not optimal_opportunities:
            logs.append("No opportunities meet the minimum composite score ratio.")
            return {"optimal_strategies": [], "position_to_exit": {}}

        # Calculate total composite score for optimal opportunities
        total_composite_score = sum(
            opportunity["composite_score"] for opportunity in optimal_opportunities
        )

        # Assign percentage of funds to each optimal opportunity
        for opportunity in optimal_opportunities:
            opportunity["funds_percentage"] = (
                opportunity["composite_score"] / total_composite_score
            ) * 100

        # Calculate relative percentages
        funds_percentages = [
            opportunity["funds_percentage"] for opportunity in optimal_opportunities
        ]
        relative_percentages = calculate_relative_percentages(funds_percentages)

        # Update opportunities with relative percentages
        for opportunity, relative_percentage in zip(optimal_opportunities, relative_percentages):
            opportunity["relative_funds_percentage"] = round(relative_percentage, 10)    
    
    return {"optimal_strategies": optimal_opportunities, "position_to_exit": position_to_exit} 

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)  # Default to 1 if not provided
    optimal_strategies = apply_risk_thresholds_and_select_optimal_strategy(**kwargs)
    optimal_strategies['logs'] = logs
    return optimal_strategies
