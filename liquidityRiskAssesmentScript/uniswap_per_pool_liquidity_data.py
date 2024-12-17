import requests
import math
from typing import Dict, Any


class UniswapLiquidityRiskAnalyzer:
    def __init__(self, api_key: str):
        self.API_KEY = api_key
        self.SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{self.API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    def get_pool_data(self, pool_id: str) -> Dict[str, Any]:
        """
        Fetch comprehensive pool data from The Graph
        """
        query = {
            "query": f"""
            {{
              pool(id: "{pool_id}") {{
                tick
                token0 {{
                  symbol
                  id
                  decimals
                  totalSupply
                }}
                token1 {{
                  symbol
                  id
                  decimals
                  totalSupply
                }}
                feeTier
                sqrtPrice
                liquidity
                totalValueLockedUSD
                volumeUSD
                txCount
                token0Price
                token1Price
              }}
            }}
            """
        }

        response = requests.post(self.SUBGRAPH_URL, json=query)
        if response.status_code == 200:
            json_response = response.json()
            if "data" in json_response and json_response["data"]["pool"]:
                return json_response["data"]["pool"]
            else:
                raise Exception(
                    f"Pool data not found or unexpected format: {json_response}"
                )
        else:
            raise Exception(
                f"Query failed with status {response.status_code}: {response.text}"
            )

    def calculate_liquidity_risk(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess liquidity risk based on multiple factors
        """
        # Liquidity concentration risk
        total_value_locked = float(pool_data["totalValueLockedUSD"])
        volume = float(pool_data["volumeUSD"])

        # Turnover ratio (higher is better liquidity)
        turnover_ratio = volume / total_value_locked if total_value_locked > 0 else 0

        # Transaction count as liquidity activity indicator
        tx_count = int(pool_data["txCount"])

        # Fee tier risk (lower fee tiers might indicate lower liquidity)
        fee_risk_map = {500: 0.3, 3000: 0.5, 10000: 0.7}
        fee_tier_risk = fee_risk_map.get(int(pool_data["feeTier"]), 1.0)

        # Price impact potential (sqrt price as proxy for potential slippage)
        sqrt_price = float(pool_data["sqrtPrice"])

        # Liquidity risk score (lower is riskier)
        liquidity_risk_score = {
            "turnover_ratio": turnover_ratio,
            "total_value_locked_usd": total_value_locked,
            "tx_count": tx_count,
            "fee_tier_risk": fee_tier_risk,
            "sqrt_price_volatility": sqrt_price,
            "overall_liquidity_risk": (
                (turnover_ratio * 0.3)
                + (total_value_locked / 1_000_000 * 0.2)
                + (tx_count / 1000 * 0.2)
                + (fee_tier_risk * 0.15)
                + (1 / sqrt_price * 0.15)
            ),
        }

        return liquidity_risk_score

    def detailed_liquidity_analysis(self, pool_id: str) -> Dict[str, Any]:
        """
        Comprehensive liquidity risk assessment
        """
        pool_data = self.get_pool_data(pool_id)
        liquidity_risk = self.calculate_liquidity_risk(pool_data)

        return {
            "pool_details": {
                "token0": pool_data["token0"]["symbol"],
                "token1": pool_data["token1"]["symbol"],
                "fee_tier": pool_data["feeTier"],
            },
            "liquidity_metrics": liquidity_risk,
        }


def main():
    # Replace with your actual API key
    API_KEY = "b8e4cf1b314c67d2a0109325046a7464"
    pool_id = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"

    try:
        analyzer = UniswapLiquidityRiskAnalyzer(API_KEY)
        analysis = analyzer.detailed_liquidity_analysis(pool_id)

        print("Liquidity Risk Analysis:")
        print(
            f"Pool: {analysis['pool_details']['token0']}/{analysis['pool_details']['token1']}"
        )
        print(f"Fee Tier: {analysis['pool_details']['fee_tier']}")
        print("\nLiquidity Metrics:")
        for key, value in analysis["liquidity_metrics"].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
