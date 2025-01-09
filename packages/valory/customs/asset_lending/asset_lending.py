import warnings
from concurrent.futures import ThreadPoolExecutor
import requests
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import logging
import pandas as pd
from functools import lru_cache
import aiohttp
import asyncio
import sys
import os
if sys.platform.startswith('win'):
    import asyncio
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Constants
REQUIRED_FIELDS = frozenset(("chains", "apr_threshold", "lending_asset", "current_pool", "coingecko_api_key"))
STURDY = 'Sturdy'
TVL_WEIGHT = 0.6
APR_WEIGHT = 0.4
TVL_PERCENTILE = 50
APR_PERCENTILE = 50
SCORE_PERCENTILE = 80
PERCENT_CONVERSION = 100
PRICE_IMPACT = 0.01
FETCH_AGGREGATOR_ENDPOINT = "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"

COINGECKO_NAME_TO_ID = {
    "weth": "weth",
    "stone": "stakestone-ether",
    "ezeth": "renzo-restaked-eth",
    "mode": "mode",
}

session = None

async def get_session():
    global session
    if session is None:
        session = aiohttp.ClientSession()
    return session

@lru_cache(maxsize=128)
async def get_coin_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        session = await get_session()
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        logging.error(f"Failed to fetch coin list: {e}")
    return []

@lru_cache(maxsize=128)
async def fetch_token_id(symbol: str) -> Optional[str]:
    symbol = symbol.lower()
    if symbol in COINGECKO_NAME_TO_ID:
        return COINGECKO_NAME_TO_ID[symbol]

    coin_list = await get_coin_list()
    for coin in coin_list:
        if coin['symbol'].lower() == symbol:
            return coin['id']
    
    logging.error(f"Failed to fetch id for coin with symbol: {symbol}")
    return None

@lru_cache(maxsize=1)
async def fetch_historical_data(limit: int = 720) -> List[Dict]:
    current_time_ms = int(datetime.now().timestamp() * 1000)
    one_month_ago_ms = current_time_ms - (30 * 24 * 60 * 60 * 1000)
    url = f"https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/getV2AggregatorHistoricalData?last_time={one_month_ago_ms}&limit={limit}"
    
    session = await get_session()
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        raise Exception("Failed to fetch historical data from STURDY API.")

def calculate_daily_returns(base_apy: float, reward_apy: float = 0) -> float:
    return np.power(1 + base_apy + reward_apy, 1/365) - 1

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0003) -> float:
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns, ddof=1) == 0:
        return np.nan
    return excess_returns.mean() / excess_returns.std(ddof=1)

async def get_sharpe_ratio_for_address(historical_data: List[Dict], address: str) -> float:
    records = []
    for entry in historical_data:
        try:
            timestamp = entry['timestamp']
            mapping = {ent.split("_")[1]: ent for ent in entry.get('doc', {}) if isinstance(ent, str) and len(ent.split("_")) >= 2}
            
            if address not in mapping:
                continue
                
            address_key = mapping[address]
            if address_key in entry.get('doc', {}):
                data = entry['doc'][address_key]
                records.append({
                    'timestamp': timestamp,
                    'base_apy': float(data.get('baseAPY', 0)),
                    'rewards_apy': float(data.get('rewardsAPY', 0))
                })
        except Exception as e:
            logging.error(f"Error processing historical data entry: {e}")
            continue

    if not records:
        return float('nan')

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    daily_df = df.groupby('date').median().reset_index()
    
    daily_df['daily_return'] = daily_df.apply(
        lambda row: calculate_daily_returns(row['base_apy'], row['rewards_apy']), 
        axis=1
    )
    return calculate_sharpe_ratio(daily_df['daily_return'].values)

@lru_cache(maxsize=1)
async def fetch_aggregators() -> List[Dict[str, Any]]:
    session = await get_session()
    try:
        async with session.get(FETCH_AGGREGATOR_ENDPOINT) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, dict) and 'errors' in result:
                    logging.error(f"REST API Errors: {result['errors']}")
                    return []
                return result if isinstance(result, list) else []
    except Exception as e:
        logging.error(f"REST API request failed: {e}")
        return []

async def filter_aggregators(
    chains: List[str], 
    apr_threshold: float, 
    aggregators: List[Dict], 
    lending_asset: str, 
    current_pool: str
) -> List[Dict[str, Any]]:
    filtered_aggregators = []
    tvl_list = []
    apr_list = []

    for aggregator in aggregators:
        try:
            if (aggregator.get("chainName") in chains and 
                aggregator.get('address') != current_pool and 
                aggregator.get("asset", {}).get("address") == lending_asset):
                
                total_apr = float(aggregator.get('apy', {}).get('total', 0))
                tvl = float(aggregator.get('tvl', 0))
                tvl_list.append(tvl)
                apr_list.append(total_apr)
                aggregator["total_apr"] = total_apr
                aggregator["tvl"] = tvl
                filtered_aggregators.append(aggregator)
        except (ValueError, TypeError) as e:
            logging.error(f"Error processing aggregator: {e}")
            continue

    if not filtered_aggregators:
        return []

    if len(filtered_aggregators) <= 5:
        return filtered_aggregators

    tvl_array = np.array(tvl_list)
    apr_array = np.array(apr_list)
    
    tvl_threshold = np.percentile(tvl_array, TVL_PERCENTILE)
    apr_threshold_val = np.percentile(apr_array, APR_PERCENTILE)

    max_tvl = np.max(tvl_array) if len(tvl_array) > 0 else 1
    max_apr = np.max(apr_array) if len(apr_array) > 0 else 1

    scored_aggregators = []
    for idx, aggregator in enumerate(filtered_aggregators):
        if tvl_array[idx] < tvl_threshold or apr_array[idx] < apr_threshold_val:
            continue

        score = TVL_WEIGHT * (tvl_array[idx] / max_tvl) + APR_WEIGHT * (apr_array[idx] / max_apr)
        aggregator["score"] = score
        scored_aggregators.append(aggregator)

    if not scored_aggregators:
        return []

    scores = np.array([agg["score"] for agg in scored_aggregators])
    score_threshold = np.percentile(scores, SCORE_PERCENTILE)
    
    filtered_scored_aggregators = [
        agg for agg in scored_aggregators 
        if agg["score"] >= score_threshold
    ]

    filtered_scored_aggregators.sort(key=lambda x: x["score"], reverse=True)
    return filtered_scored_aggregators[:10]

async def calculate_il_risk_score_for_lending(
    asset_token_1: str, 
    asset_token_2: str, 
    cg: CoinGeckoAPI
) -> float:
    if not asset_token_1 or not asset_token_2:
        return float('nan')

    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                cg.get_coin_market_chart_range_by_id,
                id=asset_token_1,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            future2 = executor.submit(
                cg.get_coin_market_chart_range_by_id,
                id=asset_token_2,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            
            prices_1 = future1.result()
            prices_2 = future2.result()

            if not prices_1.get('prices') or not prices_2.get('prices'):
                return float('nan')

    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return float('nan')

    try:
        prices_1_data = np.array([float(x[1]) for x in prices_1['prices']])
        prices_2_data = np.array([float(x[1]) for x in prices_2['prices']])
    except (ValueError, TypeError) as e:
        logging.error(f"Error processing price data: {e}")
        return float('nan')

    min_length = min(len(prices_1_data), len(prices_2_data))
    if min_length < 2:
        return float('nan')

    prices_1_data = prices_1_data[:min_length]
    prices_2_data = prices_2_data[:min_length]

    try:
        price_correlation = np.corrcoef(prices_1_data, prices_2_data)[0, 1]
        volatility_multiplier = np.sqrt(np.std(prices_1_data) * np.std(prices_2_data))

        P0 = prices_1_data[0] / prices_2_data[0]
        P1 = prices_1_data[-1] / prices_2_data[-1]
        il_impact = 1 - np.sqrt(P1 / P0) * (2 / (1 + P1 / P0))
        
        return float(il_impact * price_correlation * volatility_multiplier)
    except (ValueError, ZeroDivisionError) as e:
        logging.error(f"Error calculating IL risk score: {e}")
        return float('nan')

async def calculate_il_risk_score_for_silos(
    token0_symbol: str, 
    silos: List[Dict], 
    cg: CoinGeckoAPI
) -> float:
    if not token0_symbol or not isinstance(silos, list):
        return float('nan')

    token_id_cache = {}
    
    async def get_token_id(symbol: str) -> Optional[str]:
        if not symbol:
            return None
        symbol = symbol.lower()
        if symbol in token_id_cache:
            return token_id_cache[symbol]
        token_id = COINGECKO_NAME_TO_ID.get(symbol) or await fetch_token_id(symbol)
        token_id_cache[symbol] = token_id
        return token_id

    token_0_id = await get_token_id(token0_symbol)
    if not token_0_id:
        return float('nan')

    il_risk_scores = []
    for silo in silos:
        try:
            token_1_id = await get_token_id(silo.get('collateral', '').lower())
            if token_1_id:
                il_risk_score = await calculate_il_risk_score_for_lending(token_0_id, token_1_id, cg)
                if not np.isnan(il_risk_score):
                    normalized_il_risk_score = max(0, min(il_risk_score, 1))
                    il_risk_scores.append(normalized_il_risk_score)
        except Exception as e:
            logging.error(f"Error calculating IL risk score for silo: {e}")
            continue

    return float('nan') if not il_risk_scores else sum(il_risk_scores) / len(il_risk_scores)

def analyze_vault_liquidity(aggregator: Dict) -> Tuple[float, float]:
    try:
        tvl = float(aggregator.get('tvl', 0))
        total_assets = float(aggregator.get('totalAssets', 0))

        if tvl <= 0 or total_assets <= 0:
            return float('nan'), float('nan')

        depth_score = (np.log1p(tvl) * np.log1p(total_assets)) / (PRICE_IMPACT * 1000)
        liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score > 0 else 0
        max_position_size = 50 * (tvl * liquidity_risk_multiplier) / 100

        return depth_score, max_position_size
    except (ValueError, TypeError, ZeroDivisionError) as e:
        logging.error(f"Error analyzing vault liquidity: {e}")
        return float('nan'), float('nan')

def format_aggregator(aggregator: Dict) -> Dict[str, Any]:
    try:
        return {
            "chain": aggregator.get('chainName', ''),
            "pool_address": aggregator.get('address', ''),
            "dex_type": STURDY,
            "token0_symbol": aggregator.get('asset', {}).get('symbol', ''),
            "token0": aggregator.get('asset', {}).get('address', ''),
            "apr": aggregator.get('total_apr', 0) * PERCENT_CONVERSION,
            "whitelistedSilos": aggregator.get('whitelistedSilos', []),
            "il_risk_score": aggregator.get('il_risk_score', float('nan')),
            "sharpe_ratio": aggregator.get('sharpe_ratio', float('nan')),
            "depth_score": aggregator.get('depth_score', float('nan')),
            "max_position_size": aggregator.get('max_position_size', float('nan'))
        }
    except Exception as e:
        logging.error(f"Error formatting aggregator: {e}")
        return {}

async def get_best_opportunities(
    chains: List[str],
    apr_threshold: float,
    lending_asset: str,
    current_pool: str,
    coingecko_api_key: str
) -> List[Dict[str, Any]]:
    try:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
        
        data, historical_data = await asyncio.gather(
            fetch_aggregators(),
            fetch_historical_data()
        )
        
        if not data:
            return []
        
        filtered_aggregators = await filter_aggregators(
            chains, apr_threshold, data, lending_asset, current_pool
        )
        if not filtered_aggregators:
            return []

        async def process_aggregator(aggregator):
            try:
                silos = aggregator.get('whitelistedSilos', [])
                
                il_risk_score, sharpe_ratio = await asyncio.gather(
                    calculate_il_risk_score_for_silos(
                        aggregator['asset']['symbol'], 
                        silos, 
                        cg
                    ),
                    get_sharpe_ratio_for_address(historical_data, aggregator['address'])
                )
                
                depth_score, max_position_size = analyze_vault_liquidity(aggregator)
                
                aggregator.update({
                    'il_risk_score': il_risk_score,
                    'sharpe_ratio': sharpe_ratio,
                    'depth_score': depth_score,
                    'max_position_size': max_position_size
                })
                
                return aggregator
            except Exception as e:
                logging.error(f"Error processing aggregator: {e}")
                return None
        print(4)
        processed_aggregators = await asyncio.gather(
            *[process_aggregator(aggregator) for aggregator in filtered_aggregators]
        )
        print(5)
        return [
            format_aggregator(agg) for agg in processed_aggregators 
            if agg is not None
        ]
    except Exception as e:
        logging.error(f"Error in get_best_opportunities: {e}")
        return []

async def calculate_metrics(
    current_pool: Dict[str, Any], 
    coingecko_api_key: str, 
    **kwargs
) -> Optional[Dict[str, Any]]:
    try:
        cg = CoinGeckoAPI(demo_api_key=coingecko_api_key)
        historical_data = await fetch_historical_data()
        
        il_risk_score, sharpe_ratio = await asyncio.gather(
            calculate_il_risk_score_for_silos(
                current_pool.get("token0_symbol"), 
                current_pool.get('whitelistedSilos', []), 
                cg
            ),
            get_sharpe_ratio_for_address(historical_data, current_pool['pool_address'])
        )
        
        depth_score, max_position_size = analyze_vault_liquidity(current_pool)
        
        return {
            "il_risk_score": il_risk_score,
            "sharpe_ratio": sharpe_ratio,
            "max_position_size": max_position_size,
            "depth_score": depth_score
        }
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None

def run(*_args, **kwargs) -> Any:
    try:
        missing = REQUIRED_FIELDS - set(kwargs.keys())
        if missing:
            return {"error": f"Required kwargs {missing} were not provided."}

        get_metrics = kwargs.get('get_metrics', False)
        logging.info({get_metrics})
        kwargs = {k: v for k, v in kwargs.items() if k in REQUIRED_FIELDS}

        if get_metrics:
            return await calculate_metrics(**kwargs)
        else:
            result = await get_best_opportunities(**kwargs)
            if not result:
                return {"error": "No suitable aggregators found"}
            return result
    except Exception as e:
        logging.error(f"Error in run function: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}