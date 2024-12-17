import requests
import time

# API details
API_KEY = "b8e4cf1b314c67d2a0109325046a7464"
SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"


def fetch_pool_data():
    query = {
        "query": """
        {
          pools(first: 1000, orderBy: liquidity, orderDirection: desc) {
            id
            token0 {
              id
              symbol
              decimals
            }
            token1 {
              id
              symbol
              decimals
            }
            liquidity
          }
        }
        """
    }
    response = requests.post(SUBGRAPH_URL, json=query)
    if response.status_code == 200:
        return response.json()["data"]["pools"]
    else:
        raise Exception(
            f"Error fetching pool data: {response.status_code}, {response.text}"
        )


def fetch_24_hour_volume():
    timestamp_24h_ago = int(time.time()) - 86400
    query = {
        "query": f"""
        {{
          poolDayDatas(first: 1000, orderBy: date, orderDirection: desc, where: {{
            date_gt: {timestamp_24h_ago}
          }}) {{
            pool {{
              id
              token0 {{
                symbol
              }}
              token1 {{
                symbol
              }}
            }}
            date
            liquidity
            volumeUSD
            volumeToken0
            volumeToken1
          }}
        }}
        """
    }
    response = requests.post(SUBGRAPH_URL, json=query)
    if response.status_code == 200:
        return response.json()["data"]["poolDayDatas"]
    else:
        raise Exception(
            f"Error fetching 24-hour volume: {response.status_code}, {response.text}"
        )


def assess_liquidity_risk():
    pools = fetch_pool_data()
    volumes = fetch_24_hour_volume()

    for pool in pools:
        pool_id = pool["id"]
        liquidity = float(pool["liquidity"])
        token0 = pool["token0"]["symbol"]
        token1 = pool["token1"]["symbol"]

        # Match with 24-hour volume
        volume_data = next((v for v in volumes if v["pool"]["id"] == pool_id), None)
        if volume_data:
            volume_usd = float(volume_data["volumeUSD"])
            activity_ratio = volume_usd / liquidity if liquidity > 0 else 0
            print(
                f"Pool: {token0}-{token1}, Liquidity: {liquidity}, Volume (24h): ${volume_usd:.2f}, Activity Ratio: {activity_ratio:.4f}"
            )
        else:
            print(f"Pool: {token0}-{token1}, Liquidity: {liquidity}, No volume data.")


if __name__ == "__main__":
    assess_liquidity_risk()
