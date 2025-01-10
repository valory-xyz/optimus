import json
import requests
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

CAMPAIGN_TYPES = [1, 2]
HTTP_OK = [200, 201]
REQUIRED_FIELDS = ("chains", "apr_threshold", "protocols", "balancer_graphql_endpoints", "merkl_fetch_campaign_args", "chain_to_chain_id_mapping", "current_pool")

class DexTypes(Enum):
    """DexTypes"""
    BALANCER = "balancerPool"
    UNISWAP_V3 = "UniswapV3"

def check_missing_fields(kwargs: Dict[str, Any]) -> List[str]:
    """Check for missing fields and return them, if any."""
    missing = []
    for field in REQUIRED_FIELDS:
        if kwargs.get(field, None) is None:
            missing.append(field)
    return missing

def remove_irrelevant_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the irrelevant fields from the given kwargs."""
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in REQUIRED_FIELDS}
    return filtered_kwargs

def highest_apr_opportunity(
    balancer_graphql_endpoints: Dict[str, Any],
    merkl_fetch_campaign_args: Dict[str, Any],
    chains: List[str], 
    protocols: List[str],
    chain_to_chain_id_mapping: Dict[str, Any],
    apr_threshold: float, 
    current_pool: Optional[str]
) -> Dict[str, Any]:
    """Get the highest APR yielding opportunity over given protocols using Merkl API."""

    def fetch_all_pools() -> Dict[str, Any]:
        """Fetch all pools based on allowed chains."""
        if not chains:
            return {"error": "No chain selected for investment!"}

        chain_ids = ",".join(
            str(chain_to_chain_id_mapping[chain])
            for chain in chains
        )
        base_url = merkl_fetch_campaign_args.get("url")
        creator = merkl_fetch_campaign_args.get("creator")
        live = merkl_fetch_campaign_args.get("live", "true")

        query_params = {
            "chainIds": chain_ids,
            "creatorTag": creator,
            "live": live,
            "types": CAMPAIGN_TYPES,
        }
        api_url = f"{base_url}?{urlencode(query_params, doseq=True)}"

        response = requests.get(api_url, headers={"accept": "application/json"})

        if response.status_code not in HTTP_OK:
            return {"error": f"Could not retrieve data from url {api_url}. Status code {response.status_code}. Error Message: {response.text}"}

        try:
            result = json.loads(response.text)
            return result
        except (ValueError, TypeError) as e:
            return {"error": f"Could not parse response from api, the following error was encountered {type(e).__name__}: {e}"}

    def filter_eligible_pools(all_pools: Dict[str, Any]) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """Filter pools based on allowed assets and LP pools and sort by APR."""
        eligible_pools = []
        allowed_dex_types = [DexTypes[protocol.upper()].value for protocol in protocols]
        for chain_id, campaigns in all_pools.items():
            for campaign_list in campaigns.values():
                for campaign in campaign_list.values():
                    dex_type = campaign.get("type") or campaign.get("ammName")
                    if dex_type not in allowed_dex_types:
                        continue

                    campaign_apr = campaign.get("apr", 0)
                    if not campaign_apr or campaign_apr <= 0 or campaign_apr <= apr_threshold:
                        continue

                    campaign_pool_address = campaign.get("mainParameter")
                    if not campaign_pool_address or campaign_pool_address == current_pool:
                        continue
                    
                    chain = next(
                        (
                            k
                            for k, v in chain_to_chain_id_mapping.items()
                            if v == int(chain_id)
                        ),
                        None,
                    )
                    eligible_pools.append({
                        "dex_type": dex_type,
                        "chain": chain,
                        "campaign": campaign,
                        "apr": campaign_apr
                    })

        if not eligible_pools:
            return {"error": "No eligible pools found."}

        # Sort eligible pools by APR in descending order
        eligible_pools.sort(key=lambda x: x["apr"], reverse=True)
        return eligible_pools

    def determine_highest_apr_pool(eligible_pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the pool with the highest APR from the eligible pools."""
        for pool_info in eligible_pools:
            dex_type = pool_info["dex_type"]
            chain = pool_info["chain"]
            campaign = pool_info["campaign"]
            apr = pool_info["apr"]

            highest_apr_pool = extract_pool_info(dex_type, chain, apr, campaign)
            if "error" in highest_apr_pool:
                continue

            # Check the number of tokens for the highest APR pool if it's a Balancer pool
            if dex_type == DexTypes.BALANCER.value:
                pool_id = highest_apr_pool.get("pool_id")
                if highest_apr_pool.get("pool_type") is None:
                    continue
                tokensList = fetch_balancer_pool_info(pool_id, chain, detail="tokensList")
                if "error" in tokensList or len(tokensList) != 2:
                    continue

            return highest_apr_pool

        return {"error": "No highest APR pool found."}

    def extract_pool_info(dex_type, chain, apr, campaign) -> Dict[str, Any]:
        """Extract pool info from campaign data."""
        pool_address = campaign.get("mainParameter")
        if not pool_address:
            return {"error": "Pool address not found in campaign."}

        pool_token_dict = {}
        pool_id = None
        pool_type = None

        if dex_type == DexTypes.BALANCER.value:
            type_info = campaign.get("typeInfo", {})
            pool_id = type_info.get("poolId")
            pool_tokens = type_info.get("poolTokens", {})
            pool_token_items = list(pool_tokens.items())
            if len(pool_token_items) < 2 or any(
                token.get("symbol") is None for _, token in pool_token_items
            ):
                return {"error": "Invalid pool tokens found in campaign."}

            pool_type = fetch_balancer_pool_info(pool_id, chain, detail="poolType")
            if "error" in pool_type:
                return pool_type

            pool_token_dict = {
                "token0": pool_token_items[0][0],
                "token1": pool_token_items[1][0],
                "token0_symbol": pool_token_items[0][1].get("symbol"),
                "token1_symbol": pool_token_items[1][1].get("symbol"),
            }

        elif dex_type == DexTypes.UNISWAP_V3.value:
            pool_info = campaign.get("campaignParameters", {})
            if not pool_info:
                return {"error": "No pool tokens info present in campaign."}

            pool_token_dict = {
                "token0": pool_info.get("token0"),
                "token1": pool_info.get("token1"),
                "token0_symbol": pool_info.get("symbolToken0"),
                "token1_symbol": pool_info.get("symbolToken1"),
                "pool_fee": pool_info.get("poolFee"),
            }

        if any(v is None for v in pool_token_dict.values()):
            return {"error": "Invalid pool tokens found in campaign."}

        return {
            "dex_type": dex_type,
            "chain": chain,
            "apr": apr,
            "pool_address": pool_address,
            "pool_id": pool_id,
            "pool_type": pool_type,
            **pool_token_dict
        }

    def fetch_balancer_pool_info(pool_id: str, chain: str, detail: str) -> Dict[str, Any]:
        """Fetch the pool type for a Balancer pool using a GraphQL query."""
        query = f"""
                    query {{
                    pools(where: {{ id: "{pool_id}" }}) {{
                        id
                        {detail}
                    }}
                    }}
                """

        url = balancer_graphql_endpoints.get(chain)
        if not url:
            return {"error": f"GraphQL endpoint not found for chain {chain}."}

        response = requests.post(
            url,
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code not in HTTP_OK:
            return {"error": f"Could not retrieve data from GraphQL endpoint. Status code {response.status_code}."}

        try:
            res = json.loads(response.text)
            if res is None:
                return {"error": "No data returned from GraphQL endpoint."}

            pools = res.get("data", {}).get("pools", [])
            if pools:
                return {detail: pools[0].get(detail)}
            return {"error": "No pools found in GraphQL response."}
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON response from GraphQL endpoint."}

    all_pools = fetch_all_pools()
    if "error" in all_pools:
        return all_pools

    eligible_pools = filter_eligible_pools(all_pools)
    if "error" in eligible_pools:
        return eligible_pools

    highest_apr_pool = determine_highest_apr_pool(eligible_pools)
    return highest_apr_pool

def run(*_args, **kwargs) -> Dict[str, Union[bool, str]]:
    """Run the strategy."""
    missing = check_missing_fields(kwargs)
    if len(missing) > 0:
        return {"error": f"Required kwargs {missing} were not provided."}

    kwargs = remove_irrelevant_fields(kwargs)
    return highest_apr_opportunity(**kwargs)
