import warnings
import time
import requests
import numpy as np
import logging
import pandas as pd
import pyfolio as pf
from web3 import Web3
from typing import Dict, Union, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pycoingecko import CoinGeckoAPI
import json
import statistics

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
@dataclass
class Constants:
    SUPPORTED_POOL_TYPES = {
        "WEIGHTED": "Weighted",
        "COMPOSABLE_STABLE": "ComposableStable",
        "LIQUIDITY_BOOTSTRAPING": "LiquidityBootstrapping",
        "META_STABLE": "MetaStable",
        "STABLE": "Stable",
        "INVESTMENT": "Investment"
    }
    REQUIRED_FIELDS = frozenset({"chains", "apr_threshold", "graphql_endpoint", "current_pool", "coingecko_api_key"})
    EXCLUDED_APR_TYPES = frozenset({"IB_YIELD", "MERKL", "SWAP_FEE", "SWAP_FEE_7D", "SWAP_FEE_30D"})
    CHAIN_URLS = {
        "mode": "https://1rpc.io/mode",
        "optimism": "https://mainnet.optimism.io",
        "base": "https://1rpc.io/base"
    }
    METRICS = {
        'TVL_WEIGHT': 0.7,
        'APR_WEIGHT': 0.3,
        'TVL_PERCENTILE': 50,
        'APR_PERCENTILE': 50,
        'SCORE_PERCENTILE': 80
    }

# Session management
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class Cache:
    def __init__(self):
        self.coin_list = None
        self.web3_connections = {}
        self.token_names = {}
        self.session = create_session()

    def fetch_coin_list(self):
        if not self.coin_list:
            try:
                response = self.session.get("https://api.coingecko.com/api/v3/coins/list")
                response.raise_for_status()
                self.coin_list = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch coin list: {e}")
                self.coin_list = []
        return self.coin_list

    @lru_cache(maxsize=1000)
    def get_token_name(self, chain_name: str, token_address: str) -> Optional[str]:
        if not Web3.is_address(token_address):
            return None
            
        web3 = self.get_web3_connection(chain_name)
        if not web3:
            return None

        try:
            contract = web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=[{
                    "constant": True,
                    "inputs": [],
                    "name": "name",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                }]
            )
            return contract.functions.name().call()
        except Exception as e:
            logger.debug(f"Error fetching token name: {e}")
            return None

    def get_web3_connection(self, chain_name: str) -> Optional[Web3]:
        if chain_name not in self.web3_connections:
            chain_url = Constants.CHAIN_URLS.get(chain_name)
            if not chain_url:
                return None
            web3 = Web3(Web3.HTTPProvider(chain_url))
            if web3.is_connected():
                self.web3_connections[chain_name] = web3
            else:
                return None
        return self.web3_connections[chain_name]

class BalancerAnalyzer:
    def __init__(self, cache: Cache):
        self.cache = cache
        self.coingecko = None
        self.constants = Constants()

    def initialize_coingecko(self, api_key: str):
        if not self.coingecko:
            self.coingecko = CoinGeckoAPI(demo_api_key=api_key)

    def runns(self, **kwargs) -> Dict[str, Any]:
        missing_fields = self._check_missing_fields(kwargs)
        if missing_fields:
            return {"error": f"Required fields missing: {missing_fields}"}

        try:
            self.initialize_coingecko(kwargs["coingecko_api_key"])
            self.cache.fetch_coin_list()

            if kwargs.get('get_metrics', False):
                return self.calculate_metrics(kwargs["current_pool"])
            else:
                return self.get_opportunities(**kwargs)
        except Exception as e:
            logger.error(f"Error in runns: {e}")
            return {"error": str(e)}

    def _check_missing_fields(self, kwargs: Dict[str, Any]) -> List[str]:
        return [field for field in self.constants.REQUIRED_FIELDS if field not in kwargs]

    def get_opportunities(self, chains: List[str], apr_threshold: float, graphql_endpoint: str, 
                         current_pool: str, **kwargs) -> List[Dict[str, Any]]:
        pools = self._fetch_balancer_pools(chains, graphql_endpoint)
        if isinstance(pools, dict) and "error" in pools:
            return pools

        filtered_pools = self._filter_pools(pools, current_pool, apr_threshold)
        if not filtered_pools:
            return {"error": "No suitable pools found"}

        return self._process_pools(filtered_pools)

    def _fetch_balancer_pools(self, chains: List[str], graphql_endpoint: str) -> List[Dict[str, Any]]:
        chain_list = ', '.join(chain.upper() for chain in chains)
        query = f"""
        {{
          poolGetPools(where: {{chainIn: [{chain_list}]}} first: 100) {{
            id
            address
            chain
            type
            poolTokens {{
              address
              symbol
            }}
            dynamicData {{
              totalLiquidity
              aprItems {{
                type
                apr
              }}
            }}
          }}
        }}
        """
        try:
            response = self.cache.session.post(graphql_endpoint, json={"query": query})
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("poolGetPools", [])
        except Exception as e:
            logger.error(f"Error fetching pools: {e}")
            return {"error": str(e)}

    def _filter_pools(self, pools: List[Dict[str, Any]], current_pool: str, 
                 apr_threshold: float) -> List[Dict[str, Any]]:
    # First pass: collect all eligible pools
        qualifying_pools = []
        for pool in pools:
            if (pool.get('address') != current_pool and 
                len(pool.get('poolTokens', [])) == 2 and 
                self.constants.SUPPORTED_POOL_TYPES.get(pool.get('type'))):
                
                pool['type'] = self.constants.SUPPORTED_POOL_TYPES[pool['type']]
                pool['apr'] = self._calculate_total_apr(pool)
                pool['tvl'] = float(pool.get('dynamicData', {}).get('totalLiquidity', 0))
                qualifying_pools.append(pool)
    
        if len(qualifying_pools) <= 5:
            return qualifying_pools
    
        # Calculate thresholds
        tvl_list = [p['tvl'] for p in qualifying_pools]
        apr_list = [p['apr'] for p in qualifying_pools]
    
        tvl_threshold = np.percentile(tvl_list, self.constants.METRICS['TVL_PERCENTILE'])
        apr_threshold = max(np.percentile(apr_list, self.constants.METRICS['APR_PERCENTILE']), 
                           apr_threshold / 100)  # Convert input threshold to decimal
        max_tvl = max(tvl_list)
        max_apr = max(apr_list)
    
        # Score and filter pools
        scored_pools = []
        for pool in qualifying_pools:
            if pool['tvl'] >= tvl_threshold:  # Remove APR threshold here for initial scoring
                score = (
                    self.constants.METRICS['TVL_WEIGHT'] * (pool['tvl'] / max_tvl) +
                    self.constants.METRICS['APR_WEIGHT'] * (pool['apr'] / max_apr)
                )
                pool['score'] = score
                scored_pools.append(pool)
    
        if not scored_pools:
            return []
    
        # Sort by score and return top 10
        scored_pools.sort(key=lambda x: x['score'], reverse=True)
        return scored_pools[:10]

    def _calculate_total_apr(self, pool: Dict[str, Any]) -> float:
        apr_items = pool.get('dynamicData', {}).get('aprItems', [])
        return sum(
            item['apr'] for item in apr_items 
            if item['type'] not in self.constants.EXCLUDED_APR_TYPES
        )

    def _apply_scoring(self, pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pools:
            return []

        tvl_list = [p['tvl'] for p in pools]
        apr_list = [p['apr'] for p in pools]

        tvl_threshold = np.percentile(tvl_list, self.constants.METRICS['TVL_PERCENTILE'])
        apr_threshold = np.percentile(apr_list, self.constants.METRICS['APR_PERCENTILE'])

        max_tvl = max(tvl_list)
        max_apr = max(apr_list)

        scored_pools = []
        for pool in pools:
            if pool['tvl'] >= tvl_threshold and pool['apr'] >= apr_threshold:
                score = (
                    self.constants.METRICS['TVL_WEIGHT'] * (pool['tvl'] / max_tvl) +
                    self.constants.METRICS['APR_WEIGHT'] * (pool['apr'] / max_apr)
                )
                pool['score'] = score
                scored_pools.append(pool)

        scored_pools.sort(key=lambda x: x['score'], reverse=True)
        return scored_pools[:10]

    def _process_pools(self, pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_pools = []
        for pool in pools:
            try:
                metrics = self.calculate_metrics(pool)
                processed_pool = {
                    "dex_type": "balancerPool",
                    "chain": pool['chain'].lower(),
                    "apr": pool['apr'] * 100,
                    "pool_address": pool['address'],
                    "pool_id": pool['id'],
                    "pool_type": pool['type'],
                    "token0": pool['poolTokens'][0]['address'],
                    "token1": pool['poolTokens'][1]['address'],
                    "token0_symbol": pool['poolTokens'][0]['symbol'],
                    "token1_symbol": pool['poolTokens'][1]['symbol'],
                    **metrics
                }
                processed_pools.append(processed_pool)
            except Exception as e:
                logger.error(f"Error processing pool {pool.get('id')}: {e}")
                continue

        return processed_pools

    def calculate_metrics(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._calculate_il_risk, pool_data): "il_risk_score",
                    executor.submit(self._calculate_sharpe_ratio, pool_data): "sharpe_ratio",
                    executor.submit(self._calculate_liquidity_metrics, pool_data): "liquidity_metrics"
                }
                
                results = {}
                for future in futures.keys():
                    metric_name = futures[future]
                    try:
                        results[metric_name] = future.result()
                    except Exception as e:
                        logger.error(f"Error calculating {metric_name}: {e}")
                        results[metric_name] = float('nan')
                
                return results
        except Exception as e:
            logger.error(f"Error in calculate_metrics: {e}")
            return {
                "il_risk_score": float('nan'),
                "sharpe_ratio": float('nan'),
                "liquidity_metrics": {
                    "depth_score": float('nan'),
                    "max_position_size": float('nan')
                }
            }

    def _calculate_il_risk(self, pool_data: Dict[str, Any]) -> float:
        token0_id = self.get_token_id(pool_data['token0'], pool_data['token0_symbol'], pool_data['chain'])
        token1_id = self.get_token_id(pool_data['token1'], pool_data['token1_symbol'], pool_data['chain'])
        
        if not (token0_id and token1_id):
            return float('nan')

        to_timestamp = int(datetime.now().timestamp())
        from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        
        try:
            prices = self._fetch_price_data(token0_id, token1_id, from_timestamp, to_timestamp)
            if not prices:
                return float('nan')
            
            return self._compute_il_risk_score(*prices)
        except Exception as e:
            # logger.error(f"Error calculating IL risk: {e}")
            return float('nan')

    def _calculate_sharpe_ratio(self, pool_data: Dict[str, Any]) -> float:
        try:
            query = f"""
            {{
                poolGetSnapshots(
                    chain: {pool_data['chain'].upper()}
                    id: "{pool_data['id']}"
                    range: ONE_YEAR
                ) {{
                    timestamp
                    sharePrice
                    fees24h
                    totalLiquidity
                }}
            }}
            """
            response = self.cache.session.post("https://api-v3.balancer.fi/", json={'query': query})
            data = response.json()['data']['poolGetSnapshots']
            
            if not data:
                return float('nan')

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            df['sharePrice'] = pd.to_numeric(df['sharePrice'])
            price_returns = df['sharePrice'].pct_change()
            
            df['fees24h'] = pd.to_numeric(df['fees24h'])
            df['totalLiquidity'] = pd.to_numeric(df['totalLiquidity'])
            fee_returns = df['fees24h'] / df['totalLiquidity']
            
            total_returns = (price_returns + fee_returns).dropna()
            total_returns = total_returns.replace([np.inf, -np.inf], np.nan)
            
            return pf.timeseries.sharpe_ratio(total_returns)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return float('nan')

    def _calculate_liquidity_metrics(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        try:
            query = f"""
            {{
                poolGetSnapshots(
                    id: "{pool_data['id']}"
                    chain: {pool_data['chain'].upper()}
                    range: NINETY_DAYS
                ) {{
                    totalLiquidity
                    volume24h
                    timestamp
                }}
            }}
            """
            response = self.cache.session.post("https://api-v3.balancer.fi/", json={'query': query})
            snapshots = response.json()['data']['poolGetSnapshots']
            
            if not snapshots:
                return {"depth_score": float('nan'), "max_position_size": float('nan')}

            avg_tvl = statistics.mean(float(s['totalLiquidity']) for s in snapshots)
            avg_volume = statistics.mean(float(s.get('volume24h', 0)) for s in snapshots)

            price_impact = 0.01  # 1% price impact
            depth_score = (np.log1p(avg_tvl) * np.log1p(avg_volume)) / (price_impact * 100) if avg_tvl and avg_volume else 0
            liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
            max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100

            return {
                "depth_score": depth_score,
                "max_position_size": max_position_size
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {"depth_score": float('nan'), "max_position_size": float('nan')}

    def _fetch_price_data(self, token0_id: str, token1_id: str, 
                         from_timestamp: int, to_timestamp: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            prices1 = self.coingecko.get_coin_market_chart_range_by_id(
                id=token0_id,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            prices2 = self.coingecko.get_coin_market_chart_range_by_id(
                id=token1_id,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            
            return (
                np.array([x[1] for x in prices1['prices']]),
                np.array([x[1] for x in prices2['prices']])
            )
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    def _compute_il_risk_score(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        min_length = min(len(prices1), len(prices2))
        if min_length < 2:
            return float('nan')
            
        prices1 = prices1[:min_length]
        prices2 = prices2[:min_length]
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        volatility = np.sqrt(np.std(prices1) * np.std(prices2))
        price_ratio_start = prices1[0] / prices2[0]
        price_ratio_end = prices1[-1] / prices2[-1]
        
        il_impact = self.calculate_il_impact(price_ratio_start, price_ratio_end)
        return float(il_impact * correlation * volatility)

    def calculate_il_impact(self, price_ratio_start: float, price_ratio_end: float) -> float:
        return 1 - np.sqrt(price_ratio_end / price_ratio_start) * (2 / (1 + price_ratio_end / price_ratio_start))

    def get_token_id(self, token_address: str, symbol: str, chain_name: str) -> Optional[str]:
        symbol = symbol.lower()
        matching_coins = [
            coin for coin in self.cache.coin_list 
            if coin['symbol'].lower() == symbol
        ]

        if len(matching_coins) == 1:
            return matching_coins[0]['id']

        token_name = self.cache.get_token_name(chain_name, token_address)
        if not token_name:
            return None

        normalized_name = token_name.lower().replace(" ", "")
        for coin in matching_coins:
            if coin['name'].lower().replace(" ", "") == normalized_name:
                return coin['id']
        return None

def run(*_args, **kwargs) -> Any:
    # Initialize cache and analyzer
    cache = Cache()
    analyzer = BalancerAnalyzer(cache)
    
    try:
        result = analyzer.runns(**kwargs)
        return result
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}    
