# test_uniswap_v3_fork_e2e.py
# Example fork-based integration test for Uniswap v3
# Usage:
#   1) Run anvil fork first (replace YOUR_KEY):
#      anvil --fork-url https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY \
#            --fork-block-number 20850000 --chain-id 1 --port 8545 --no-rate-limit
#   2) pytest -q test_uniswap_v3_fork_e2e.py

import json, time
from web3 import Web3

# Minimal ABIs (only methods/events we use)
POSITION_MANAGER_ABI = [
  {"anonymous":False,"inputs":[
      {"indexed":True,"internalType":"uint256","name":"tokenId","type":"uint256"},
      {"indexed":False,"internalType":"uint128","name":"liquidity","type":"uint128"},
      {"indexed":False,"internalType":"uint256","name":"amount0","type":"uint256"},
      {"indexed":False,"internalType":"uint256","name":"amount1","type":"uint256"}
    ],"name":"IncreaseLiquidity","type":"event"},
  {"inputs":[{"components":[
        {"internalType":"address","name":"token0","type":"address"},
        {"internalType":"address","name":"token1","type":"address"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"int24","name":"tickLower","type":"int24"},
        {"internalType":"int24","name":"tickUpper","type":"int24"},
        {"internalType":"uint256","name":"amount0Desired","type":"uint256"},
        {"internalType":"uint256","name":"amount1Desired","type":"uint256"},
        {"internalType":"uint256","name":"amount0Min","type":"uint256"},
        {"internalType":"uint256","name":"amount1Min","type":"uint256"},
        {"internalType":"address","name":"recipient","type":"address"},
        {"internalType":"uint256","name":"deadline","type":"uint256"}
    ],"internalType":"struct INonfungiblePositionManager.MintParams","name":"params","type":"tuple"}],
    "name":"mint","outputs":[
      {"internalType":"uint256","name":"tokenId","type":"uint256"},
      {"internalType":"uint128","name":"liquidity","type":"uint128"},
      {"internalType":"uint256","name":"amount0","type":"uint256"},
      {"internalType":"uint256","name":"amount1","type":"uint256"}],
    "stateMutability":"payable","type":"function"},
  {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],
    "name":"positions",
    "outputs":[
      {"internalType":"uint96","name":"nonce","type":"uint96"},
      {"internalType":"address","name":"operator","type":"address"},
      {"internalType":"address","name":"token0","type":"address"},
      {"internalType":"address","name":"token1","type":"address"},
      {"internalType":"uint24","name":"fee","type":"uint24"},
      {"internalType":"int24","name":"tickLower","type":"int24"},
      {"internalType":"int24","name":"tickUpper","type":"int24"},
      {"internalType":"uint128","name":"liquidity","type":"uint128"},
      {"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},
      {"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},
      {"internalType":"uint128","name":"tokensOwed0","type":"uint128"},
      {"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],
    "stateMutability":"view","type":"function"},
  {"inputs":[{"components":[
      {"internalType":"uint256","name":"tokenId","type":"uint256"},
      {"internalType":"uint128","name":"liquidity","type":"uint128"},
      {"internalType":"uint256","name":"amount0Min","type":"uint256"},
      {"internalType":"uint256","name":"amount1Min","type":"uint256"},
      {"internalType":"uint256","name":"deadline","type":"uint256"}],
    "internalType":"struct INonfungiblePositionManager.DecreaseLiquidityParams","name":"params","type":"tuple"}],
    "name":"decreaseLiquidity","outputs":[
      {"internalType":"uint256","name":"amount0","type":"uint256"},
      {"internalType":"uint256","name":"amount1","type":"uint256"}],
    "stateMutability":"payable","type":"function"}
]

POOL_ABI = [
  {"inputs":[],"name":"slot0","outputs":[
      {"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},
      {"internalType":"int24","name":"tick","type":"int24"},
      {"internalType":"uint16","name":"observationIndex","type":"uint16"},
      {"internalType":"uint16","name":"observationCardinality","type":"uint16"},
      {"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},
      {"internalType":"uint8","name":"feeProtocol","type":"uint8"},
      {"internalType":"bool","name":"unlocked","type":"bool"}],
    "stateMutability":"view","type":"function"}
]

ERC20_ABI = [
  {"constant":False,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
  {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"}
]

POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
WETH9 = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
DAI = Web3.to_checksum_address("0x6B175474E89094C44Da98b954EedeAC495271d0F")
POOL_DAI_WETH_0_3 = Web3.to_checksum_address("0xC2e9F25be6257c210d7ADf0d4Cd6E3E881ba25f8")
BINANCE_HOT_WALLET = Web3.to_checksum_address("0x28C6c06298d514Db089934071355E5743bf21d60")

def impersonate(w3, addr):
    w3.provider.make_request("anvil_impersonateAccount", [addr])

def stop_impersonate(w3, addr):
    w3.provider.make_request("anvil_stopImpersonatingAccount", [addr])

def test_uniswap_v3_mint_and_decrease_liquidity(w3):
    whale = BINANCE_HOT_WALLET
    impersonate(w3, whale)

    pm = w3.eth.contract(address=POSITION_MANAGER, abi=POSITION_MANAGER_ABI)
    pool = w3.eth.contract(address=POOL_DAI_WETH_0_3, abi=POOL_ABI)
    dai = w3.eth.contract(address=DAI, abi=ERC20_ABI)
    weth = w3.eth.contract(address=WETH9, abi=ERC20_ABI)

    sqrtP, tick, *_ = pool.functions.slot0().call()
    print(f"Current tick={tick}, sqrtPriceX96={sqrtP}")

    # Approve position manager
    dai.functions.approve(POSITION_MANAGER, 2**256 - 1).transact({"from": whale}) # just for example, In real tests, always calculate
    weth.functions.approve(POSITION_MANAGER, 2**256 - 1).transact({"from": whale}) # just for example, In real tests, always calculate

    # Mint a narrow range around the current price
    deadline = int(time.time()) + 600
    tx = pm.functions.mint({
        "token0": DAI,
        "token1": WETH9,
        "fee": 3000,
        "tickLower": tick - 60,
        "tickUpper": tick + 60,
        "amount0Desired": 1000 * 10**18,
        "amount1Desired": 1 * 10**18,
        "amount0Min": 0, # just for example, In real tests, always calculate based on slippage
        "amount1Min": 0, # just for example, In real tests, always calculate based on slippage
        "recipient": whale,
        "deadline": deadline,
    }).transact({"from": whale, "value": 0})
    r = w3.eth.wait_for_transaction_receipt(tx)

    # Verify IncreaseLiquidity event
    events = pm.events.IncreaseLiquidity().process_receipt(r)
    assert events, "IncreaseLiquidity not emitted"

    token_id = events[0]["args"]["tokenId"]
    pos = pm.functions.positions(token_id).call()
    liquidity = pos[7]
    assert liquidity > 0, "Minted liquidity must be > 0"

    # Decrease half of liquidity
    pm.functions.decreaseLiquidity({
        "tokenId": token_id,
        "liquidity": liquidity, 
        "amount0Min": 0, # just for example, In real tests, always calculate based on slippage
        "amount1Min": 0, # just for example, In real tests, always calculate based on slippage
        "deadline": int(time.time()) + 600,
    }).transact({"from": whale})

    stop_impersonate(w3, whale)
