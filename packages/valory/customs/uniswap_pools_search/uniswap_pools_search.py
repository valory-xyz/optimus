import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import pyfolio as pf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
import logging
from pycoingecko import CoinGeckoAPI

logging.basicConfig(level=logging.INFO)

# Constants
DAYS_IN_YEAR = 365
PERCENT_CONVERSION = 100
FEE_RATE_DIVISOR = 1000000
PRICE_IMPACT = 0.01
MAX_POSITION_BASE = 50

class PoolAnalyzer:
    def __init__(self, chains: List[str], graphql_endpoints: Dict[str, str], 
                 current_pool: str, apr_threshold: float, coingecko_api_key: str):
        self.chains = chains
        self.graphql_endpoints = graphql_endpoints
        self.current_pool = current_pool
        self.apr_threshold = apr_threshold
        self.coingecko_api_key = coingecko_api_key
        self.cg = CoinGeckoAPI(api_key=coingecko_api_key)
        self.coin_list = self._fetch_coin_list()

    @staticmethod
    def _fetch_coin_list():
        try:
            response = requests.get("https://api.coingecko.com/api/v3/coins/list")
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return []

    def _get_token_id(self, token_address: str, symbol: str) -> Optional[str]:
        matching_coins = [c for c in self.coin_list if c['symbol'].lower() == symbol.lower()]
        if not matching_coins:
            return None
        if len(matching_coins) == 1:
            return matching_coins[0]['id']
        
        # If multiple matches, try to find the best match
        for coin in matching_coins:
            if coin['name'].lower() == symbol.lower():
                return coin['id']
        return matching_coins[0]['id']  # Return first match if no better match found

    def _calculate_il_risk_score(self, token0_id: str, token1_id: str) -> float:
        try:
            to_timestamp = int(datetime.now().timestamp())
            from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())

            # Fetch historical price data
            prices_1 = self.cg.get_coin_market_chart_range_by_id(
                id=token0_id,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            prices_2 = self.cg.get_coin_market_chart_range_by_id(
                id=token1_id,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )

            # Extract price data
            prices_1_data = np.array([x[1] for x in prices_1['prices']])
            prices_2_data = np.array([x[1] for x in prices_2['prices']])

            # Ensure equal length
            min_length = min(len(prices_1_data), len(prices_2_data))
            prices_1_data = prices_1_data[:min_length]
            prices_2_data = prices_2_data[:min_length]

            # Calculate metrics
            price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
            volatility_1 = np.std(prices_1_data)
            volatility_2 = np.std(prices_2_data)
            volatility_multiplier = np.sqrt(volatility_1 * volatility_2)

            # Calculate IL impact
            P0 = prices_1_data[0] / prices_2_data[0]
            P1 = prices_1_data[-1] / prices_2_data[-1]
            il_impact = 1 - np.sqrt(P1 / P0) * (2 / (1 + P1 / P0))

            return float(il_impact * price_correlation * volatility_multiplier)
        except Exception as e:
            logging.error(f"Error calculating IL risk score: {e}")
            return float('nan')

    def _run_query(self, query: str, endpoint: str) -> Dict:
        try:
            response = requests.post(
                endpoint,
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            )
            return response.json().get('data', {})
        except requests.RequestException:
            return {}

    def _calculate_apr(self, daily_volume: float, tvl: float, fee_rate: float) -> float:
        return 0 if tvl == 0 else (daily_volume / tvl) * fee_rate * DAYS_IN_YEAR * PERCENT_CONVERSION

    def _get_pool_metrics(self, pool: Dict) -> Dict:
        chain = pool['chain'].lower()
        endpoint = self.graphql_endpoints.get(chain)
        
        # Get token IDs for IL calculation
        token0_id = self._get_token_id(pool['token0']['id'], pool['token0']['symbol'])
        token1_id = self._get_token_id(pool['token1']['id'], pool['token1']['symbol'])
        
        # Calculate IL risk score if token IDs are available
        il_risk_score = self._calculate_il_risk_score(token0_id, token1_id) if token0_id and token1_id else float('nan')
        
        # Calculate other metrics
        sharpe_ratio = self._calculate_sharpe_ratio(pool['id'], endpoint)
        depth_score, max_position = self._assess_liquidity(pool['id'], endpoint)
        
        return {
            "dex_type": "UniswapV3",
            "chain": chain,
            "apr": pool['apr'],
            "pool_address": pool['id'],
            "token0": pool['token0']['id'],
            "token1": pool['token1']['id'],
            "token0_symbol": pool['token0']['symbol'],
            "token1_symbol": pool['token1']['symbol'],
            "il_risk_score": il_risk_score,
            "sharpe_ratio": sharpe_ratio,
            "depth_score": depth_score,
            "max_position_size": max_position
        }

    def calculate_single_pool_metrics(self, pool_data: Dict) -> Dict:
        """Calculate metrics for a single pool."""
        token0_id = self._get_token_id(pool_data['token0'], pool_data['token0_symbol'])
        token1_id = self._get_token_id(pool_data['token1'], pool_data['token1_symbol'])
        
        il_risk_score = self._calculate_il_risk_score(token0_id, token1_id) if token0_id and token1_id else float('nan')
        sharpe_ratio = self._calculate_sharpe_ratio(pool_data['pool_address'], self.graphql_endpoints[pool_data['chain']])
        depth_score, max_position_size = self._assess_liquidity(pool_data['pool_address'], self.graphql_endpoints[pool_data['chain']])
        
        return {
            "il_risk_score": il_risk_score,
            "sharpe_ratio": sharpe_ratio,
            "depth_score": depth_score,
            "max_position_size": max_position_size
        }

    def _calculate_sharpe_ratio(self, pool_address: str, endpoint: str) -> float:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        query = f"""
        {{
          poolDayDatas(
            where: {{
              pool: "{pool_address.lower()}"
              date_gt: {int(start_date.timestamp())}
            }}
            orderBy: date
            orderDirection: asc
          ) {{
            date
            tvlUSD
            feesUSD
          }}
        }}
        """
        data = self._run_query(query, endpoint).get('poolDayDatas', [])
        if not data:
            return float('nan')
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
        df['total_value'] = pd.to_numeric(df['tvlUSD']) + pd.to_numeric(df['feesUSD'])
        returns = df.set_index('date')['total_value'].pct_change().dropna()
        
        return float(pf.timeseries.sharpe_ratio(returns))

    def _assess_liquidity(self, pool_id: str, endpoint: str) -> Tuple[float, float]:
        query = f"""
        {{
          pool(id: "{pool_id.lower()}") {{
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
          }}
        }}
        """
        pool_data = self._run_query(query, endpoint).get('pool', {})
        if not pool_data:
            return float('nan'), float('nan')

        tvl_token0 = float(pool_data.get('totalValueLockedToken0', 0))
        tvl_token1 = float(pool_data.get('totalValueLockedToken1', 0))
        
        if tvl_token0 <= 0 or tvl_token1 <= 0:
            return float('nan'), float('nan')
            
        depth_score = (np.log1p(tvl_token0) * np.log1p(tvl_token1)) / (PRICE_IMPACT * 100)
        liquidity_multiplier = max(0, 1 - (1 / depth_score))
        max_position = MAX_POSITION_BASE * (float(pool_data.get('totalValueLockedUSD', 0)) * liquidity_multiplier) / 100
        
        return depth_score, max_position

    def analyze_pools(self) -> List[Dict]:
        pools_query = """
        {
          pools(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id
            feeTier
            liquidity
            volumeUSD
            totalValueLockedUSD
            token0 { id symbol }
            token1 { id symbol }
          }
        }
        """
        
        all_pools = []
        for chain in self.chains:
            endpoint = self.graphql_endpoints.get(chain)
            if not endpoint:
                continue
                
            pools = self._run_query(pools_query, endpoint).get('pools', [])
            for pool in pools:
                fee_rate = float(pool['feeTier']) / FEE_RATE_DIVISOR
                tvl = float(pool['totalValueLockedUSD'])
                daily_volume = float(pool['volumeUSD'])
                
                pool['chain'] = chain
                pool['apr'] = self._calculate_apr(daily_volume, tvl, fee_rate)
                
                if pool['id'] != self.current_pool and tvl > 0:
                    all_pools.append(pool)

        # Sort pools by APR and get top 10
        all_pools.sort(key=lambda x: x['apr'], reverse=True)
        top_pools = all_pools[:10]

        return [self._get_pool_metrics(pool) for pool in top_pools]

def main():
    # Example usage
    kwargs = {
        "chains": ["ethereum"],
        "apr_threshold": 0.05,
        "graphql_endpoints": {
            "ethereum": "https://gateway.thegraph.com/api/b8e4cf1b314c67d2a0109325046a7464/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
        },
        "current_pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "coingecko_api_key": "CG-mf5xZnGELpSXeSqmHDLY2nNU",
    }
    
    analyzer = PoolAnalyzer(**kwargs)
    
    # To get metrics for a single pool
    current_pool_data = {
        "chain": "ethereum",
        "pool_address": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "token0": "0x6b175474e89094c44da98b954eedeac495271d0f",
        "token1": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "token0_symbol": "DAI",
        "token1_symbol": "WETH"
    }
    
    single_pool_metrics = analyzer.calculate_single_pool_metrics(current_pool_data)
    print("\nSingle Pool Metrics:")
    print(single_pool_metrics)
    
    # To get analysis of all pools
    print("\nAll Pools Analysis:")
    results = analyzer.analyze_pools()
    for pool in results:
        print(pool)
