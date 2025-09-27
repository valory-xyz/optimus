# Fix for Safe API 503 errors
# Replace the request_with_retries function with this improved version:

def request_with_retries(
    endpoint: str,
    params: Dict = None,
    headers: Dict = None,
    method: str = "GET",
    body: Dict = None,
    rate_limited_code: int = 429,
    retry_wait: int = 30,  # Reduced from 60
    max_retries: int = 5   # Increased from 3
) -> Tuple[bool, Dict]:
    for attempt in range(max_retries):
        try:
            if method.upper() == "POST":
                cache_key = f"POST_{endpoint}_{str(body or {})}"
                cached_response = get_cached_request(cache_key)
                if cached_response is not None:
                    return len(cached_response) > 0, cached_response
                
                response = requests.post(endpoint, headers=headers, json=body)

                if response.ok:
                    update_request_cache(cache_key, response.json())
            else:
                # Check cache first for GET requests
                cache_key = f"{endpoint}_{str(params or {})}"
                cached_response = get_cached_request(cache_key)
                if cached_response is not None:
                    return len(cached_response) > 0, cached_response

                response = requests.get(endpoint, headers=headers, params=params or {})

                # Cache successful responses
                if response.status_code == 200:
                    update_request_cache(cache_key, response.json())
                elif response.status_code == 404:
                    update_request_cache(cache_key, {})
            
            # Handle rate limiting and service unavailable errors with exponential backoff
            if response.status_code in [rate_limited_code, 503, 502, 504]:
                wait_time = retry_wait * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Service unavailable (status {response.status_code}). Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                logger.error(f"Request failed with status {response.status_code}")
                return False, {}
                
            return True, response.json()
            
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_wait * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
                continue
            return False, {}
    
    return False, {}

# Also add this helper function to handle Safe API failures more gracefully:

def fetch_optimism_outgoing_transfers_with_fallback(
    address: str,
    final_timestamp: int,
) -> Dict:
    """Fetch outgoing transfers with better error handling and fallback."""
    all_transfers = {}

    if not address:
        logger.warning("No address provided for fetching Optimism outgoing transfers")
        return all_transfers

    try:
        base_url = "https://safe-transaction-optimism.safe.global/api/v1"
        transfers_url = f"{base_url}/safes/{address}/transfers/"

        processed_count = 0
        page_count = 0
        max_pages = 10  # Prevent infinite loops
        consecutive_failures = 0
        max_consecutive_failures = 3

        while page_count < max_pages:
            page_count += 1

            success, response_json = request_with_retries(
                endpoint=transfers_url,
                headers={"Accept": "application/json"},
                rate_limited_code=429,
                retry_wait=10,  # Shorter wait time
                max_retries=3,  # Fewer retries per request
            )

            if not success:
                consecutive_failures += 1
                logger.error(
                    f"Failed to fetch Optimism outgoing transfers for {address} on page {page_count} (failure {consecutive_failures})"
                )
                
                # If we have too many consecutive failures, give up
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), giving up for address {address}")
                    break
                    
                # Wait longer before next attempt
                time.sleep(30)
                continue
            else:
                consecutive_failures = 0  # Reset on success

            transfers = response_json.get("results", [])
            if not transfers:
                break

            for transfer in transfers:
                # Parse timestamp
                timestamp = transfer.get("executionDate")
                if not timestamp:
                    continue

                try:
                    tx_datetime = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    tx_date = tx_datetime.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid timestamp format: {timestamp}")
                    continue

                if tx_datetime.timestamp() > final_timestamp:
                    continue

                # Only process outgoing transfers
                if transfer.get("from", "").lower() == address.lower():
                    transfer_type = transfer.get("type", "")

                    if transfer_type in ["ETHER_TRANSFER", "ERC20_TRANSFER"]:
                        try:
                            token_info = transfer.get("tokenInfo", {}) or {}
                            symbol = token_info.get("symbol", "ETH" if transfer_type == "ETHER_TRANSFER" else "")
                            decimals = int(token_info.get("decimals", 18) or 18)
                            token_address = token_info.get("address", "")
                            value = int(transfer.get("value", "0") or "0")
                            amount = value / 10**decimals

                            if amount <= 0:
                                continue
                        except (ValueError, TypeError):
                            continue

                        transfer_data = {
                            "from_address": address,
                            "to_address": transfer.get("to"),
                            "amount": amount,
                            "token_address": token_address,
                            "symbol": symbol,
                            "timestamp": timestamp,
                            "tx_hash": transfer.get("transactionHash", ""),
                            "type": transfer_type,
                        }

                        if tx_date not in all_transfers:
                            all_transfers[tx_date] = []
                        all_transfers[tx_date].append(transfer_data)
                        processed_count += 1

            # Check for next page
            cursor = response_json.get("next")
            if not cursor:
                break
            else:
                transfers_url = cursor

        logger.info(f"Completed Optimism outgoing transfers: {processed_count} found")
        return all_transfers

    except Exception as e:
        logger.error(f"Error fetching Optimism outgoing transfers: {e}")
        return {}
