import requests
import time
import json
import math

# API details
API_KEY = "b8e4cf1b314c67d2a0109325046a7464"  # Replace with your API key
SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Specific pool ID to analyze
POOL_ID = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"

# Constants
PRICE_IMPACT = 0.01  # 1% standard price impact
MAX_POSITION_BASE = 50  # Base for maximum position calculation


def fetch_pool_data(pool_id):
    query = {
        "query": f"""
        {{
          pool(id: "{pool_id.lower()}") {{
            id
            token0 {{
              id
              symbol
              decimals
            }}
            token1 {{
              id
              symbol
              decimals
            }}
            liquidity
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
          }}
        }}
        """
    }
    try:
        print(f"Fetching data for pool ID: {pool_id}")
        response = requests.post(
            SUBGRAPH_URL,
            json=query,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        print(f"Response Status Code: {response.status_code}")

        # Print full response for debugging
        response_json = response.json()
        print("Full Response:")
        print(json.dumps(response_json, indent=2))

        if response.status_code == 200:
            # Check for GraphQL errors
            if "errors" in response_json:
                print("GraphQL Errors:")
                print(json.dumps(response_json["errors"], indent=2))
                return None

            # Check for valid pool data
            data = response_json.get("data", {})
            pool = data.get("pool")

            if pool is None:
                print("No pool data found for the given ID")
                return None

            return pool
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response Text: {response.text}")
            return None

    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None


def fetch_24_hour_volume(pool_id):
    timestamp_24h_ago = int(time.time()) - 86400
    query = {
        "query": f"""
        {{
          poolDayDatas(first: 1, orderBy: date, orderDirection: desc, where: {{
            pool: "{pool_id.lower()}",
            date_gt: {timestamp_24h_ago}
          }}) {{
            date
            liquidity
            volumeUSD
            volumeToken0
            volumeToken1
          }}
        }}
        """
    }
    try:
        response = requests.post(
            SUBGRAPH_URL,
            json=query,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            pool_day_datas = data.get("data", {}).get("poolDayDatas", [])

            if not pool_day_datas:
                print("No volume data found for the pool")
                return []

            return pool_day_datas
        else:
            print(f"Error fetching 24-hour volume: {response.status_code}")
            print(f"Response Text: {response.text}")
            return []

    except Exception as e:
        print(f"Exception in volume fetch: {e}")
        return []


def calculate_metrics(pool_data, volume_data):
    try:
        # Default values to handle potential missing data
        liquidity = float(pool_data.get("liquidity", 0))
        tvl = float(pool_data.get("totalValueLockedUSD", 0))

        # Use total value locked for tokens instead of reserves
        tvl_token0 = float(pool_data.get("totalValueLockedToken0", 0))
        tvl_token1 = float(pool_data.get("totalValueLockedToken1", 0))

        volume_usd = float(volume_data[0]["volumeUSD"]) if volume_data else 0
        token0 = pool_data.get("token0", {}).get("symbol", "Token0")
        token1 = pool_data.get("token1", {}).get("symbol", "Token1")

        # Depth Score Calculation (using TVL of tokens)
        depth_score = (
            (tvl_token0 * tvl_token1) / (PRICE_IMPACT * 100)
            if tvl_token0 > 0 and tvl_token1 > 0
            else 0
        )

        # Liquidity Risk Multiplier
        liquidity_risk_multiplier = (
            max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
        )

        # Maximum Position Size
        max_position_size = MAX_POSITION_BASE * (tvl * liquidity_risk_multiplier) / 100

        # Print Detailed Metrics
        print(f"\n===== Pool Analysis: {token0}-{token1} =====")
        print(f"Total Value Locked: ${tvl:,.2f}")
        print(f"Total Value Locked Token0: {tvl_token0:,.4f} {token0}")
        print(f"Total Value Locked Token1: {tvl_token1:,.4f} {token1}")
        print(f"24h Volume: ${volume_usd:,.2f}")
        print(f"\nDepth Score: {depth_score:,.4f}")
        print(f"Liquidity Risk Multiplier: {liquidity_risk_multiplier:.4f}")
        print(f"Maximum Position Size: ${max_position_size:,.2f}")

        return {
            "token_pair": f"{token0}-{token1}",
            "tvl": tvl,
            "tvl_token0": tvl_token0,
            "tvl_token1": tvl_token1,
            "volume_24h": volume_usd,
            "depth_score": depth_score,
            "liquidity_risk_multiplier": liquidity_risk_multiplier,
            "max_position_size": max_position_size,
        }

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def assess_pool_liquidity(pool_id):
    # Fetch pool data
    pool_data = fetch_pool_data(pool_id)

    # Add explicit check for None
    if pool_data is None:
        print(f"Could not retrieve data for pool {pool_id}")
        return None

    try:
        # Fetch volume data
        volumes = fetch_24_hour_volume(pool_id)

        # Calculate and return metrics
        return calculate_metrics(pool_data, volumes)

    except Exception as e:
        print(f"Error processing pool data: {e}")
        return None


def main():
    # Analyze the specified pool
    pool_metrics = assess_pool_liquidity(POOL_ID)


if __name__ == "__main__":
    main()
