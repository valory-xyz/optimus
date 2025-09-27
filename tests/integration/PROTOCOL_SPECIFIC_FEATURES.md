# Protocol-Specific Features and Testing

This document outlines how each DeFi protocol's unique mechanics are handled in our integration tests.

## 🏦 **Balancer Protocol**

### **Unique Features**
- **Vault Architecture**: All operations go through the Balancer Vault
- **Pool Types**: Weighted, Stable, Composable Stable, Meta Stable
- **Join/Exit Kinds**: Different join/exit strategies for different pool types
- **BPT Tokens**: Balancer Pool Tokens represent liquidity positions

### **Protocol-Specific Tests**
```python
# Vault interactions
def test_vault_join_pool_transaction_encoding()
def test_vault_exit_pool_transaction_encoding()
def test_vault_get_pool_tokens_query()

# Pool type specific logic
def test_join_kind_enum_validation()
def test_exit_kind_enum_validation()
def test_pool_type_detection()

# BPT token handling
def test_user_share_calculation_isolated()
def test_pool_share_proportional_math()
```

### **Contract Interactions**
- `VaultContract.join_pool()` - Add liquidity through vault
- `VaultContract.exit_pool()` - Remove liquidity through vault
- `WeightedPoolContract.get_balance()` - Query BPT balance
- `WeightedPoolContract.get_total_supply()` - Query total BPT supply

---

## 🔄 **Uniswap V3 Protocol**

### **Unique Features**
- **Concentrated Liquidity**: Liquidity provided within specific price ranges
- **NFT Positions**: Each position is represented by an NFT token
- **Tick Math**: Complex mathematical calculations for price ranges
- **Fee Tiers**: Different fee levels (0.05%, 0.3%, 1%)
- **Position Management**: Mint, increase, decrease, collect, burn operations

### **Protocol-Specific Tests**
```python
# Tick math and concentrated liquidity
def test_tick_math_calculations()
def test_tick_range_calculation()
def test_liquidity_amount_calculation()
def test_capital_efficiency_calculation()

# NFT position management
def test_position_liquidity_calculation()
def test_nft_token_handling()
def test_position_management_logic()

# Fee collection
def test_fee_collection_calculation()
def test_fee_growth_calculation()
```

### **Contract Interactions**
- `UniswapV3NonfungiblePositionManagerContract.mint()` - Create new position
- `UniswapV3NonfungiblePositionManagerContract.decrease_liquidity()` - Reduce position
- `UniswapV3NonfungiblePositionManagerContract.collect()` - Collect fees
- `UniswapV3PoolContract.slot0()` - Get current pool state

---

## 🏎️ **Velodrome Protocol**

### **Unique Features**
- **Pool Types**: Stable pools, Volatile pools, CL pools
- **Gauge Staking**: LP tokens can be staked in gauges for additional rewards
- **Voter Contract**: Governance and gauge voting
- **Dual Rewards**: Trading fees + gauge rewards
- **CL Pools**: Concentrated liquidity similar to Uniswap V3

### **Protocol-Specific Tests**
```python
# Pool type detection
def test_velodrome_pool_type_detection()
def test_velodrome_cl_pool_detection()

# Gauge staking mechanics
def test_velodrome_gauge_staking_logic()
def test_velodrome_reward_calculation()
def test_velodrome_gauge_rewards_apr()

# CL pool specific logic
def test_velodrome_cl_pool_logic()
def test_velodrome_cl_pool_apr_calculation()
```

### **Contract Interactions**
- `VelodromePoolContract.add_liquidity()` - Add liquidity to pool
- `VelodromeGaugeContract.deposit()` - Stake LP tokens in gauge
- `VelodromeGaugeContract.get_reward()` - Claim gauge rewards
- `VelodromeVoterContract.get_gauges()` - Query gauge addresses

---

## 🔧 **Protocol-Specific Test Data**

### **Balancer Test Data**
```python
# Different pool types
balancer_weighted_pool_data = {
    "pool_type": "Weighted",
    "weights": [500000000000000000, 500000000000000000],
    "join_kind": "EXACT_TOKENS_IN_FOR_BPT_OUT"
}

balancer_stable_pool_data = {
    "pool_type": "ComposableStable", 
    "weights": [333333333333333333, 333333333333333333, 333333333333333334],
    "join_kind": "EXACT_TOKENS_IN_FOR_BPT_OUT"
}
```

### **Uniswap V3 Test Data**
```python
# Different fee tiers
uniswap_v3_pool_data = {
    "fee": 3000,  # 0.3%
    "tick_spacing": 60,
    "current_tick": -276310
}

uniswap_v3_high_fee_pool_data = {
    "fee": 10000,  # 1%
    "tick_spacing": 200,
    "current_tick": -276310
}
```

### **Velodrome Test Data**
```python
# Different pool types
velodrome_stable_pool_data = {
    "is_stable": True,
    "is_cl_pool": False,
    "gauge_address": "0xStableGauge"
}

velodrome_cl_pool_data = {
    "is_stable": False,
    "is_cl_pool": True,
    "gauge_address": "0xCLGauge"
}
```

---

## 🎯 **Protocol-Specific Workflows**

### **Balancer Workflow**
1. **Pool Selection** → Analyze pool types and weights
2. **Join Pool** → Use vault to add liquidity
3. **Track BPT** → Monitor Balancer Pool Token balance
4. **Exit Pool** → Use vault to remove liquidity

### **Uniswap V3 Workflow**
1. **Pool Analysis** → Check fee tier and current tick
2. **Position Creation** → Mint NFT position with specific range
3. **Fee Collection** → Collect accumulated fees
4. **Position Management** → Increase/decrease liquidity as needed

### **Velodrome Workflow**
1. **Pool Selection** → Choose between stable/volatile/CL pools
2. **Add Liquidity** → Provide liquidity to pool
3. **Gauge Staking** → Stake LP tokens in gauge for rewards
4. **Reward Claiming** → Claim trading fees + gauge rewards

---

## 🔍 **Testing Protocol Differences**

### **Contract Interaction Patterns**

| Protocol | Primary Contract | Secondary Contracts | Unique Operations |
|----------|------------------|---------------------|-------------------|
| **Balancer** | Vault | WeightedPool | joinPool, exitPool |
| **Uniswap V3** | PositionManager | Pool | mint, decreaseLiquidity, collect |
| **Velodrome** | Pool + Gauge | Voter | addLiquidity, deposit, getReward |

### **Yield Calculation Differences**

| Protocol | Yield Sources | Calculation Method |
|----------|---------------|-------------------|
| **Balancer** | Trading fees only | `(daily_fees * 365 / pool_tvl) * 100` |
| **Uniswap V3** | Trading fees + concentrated liquidity | `fee_apr + capital_efficiency_bonus` |
| **Velodrome** | Trading fees + gauge rewards | `trading_apr + gauge_apr` |

### **Position Management Differences**

| Protocol | Position Representation | Management Operations |
|----------|------------------------|----------------------|
| **Balancer** | BPT tokens | Join/exit pool |
| **Uniswap V3** | NFT tokens | Mint/collect/burn positions |
| **Velodrome** | LP tokens + gauge stakes | Add/remove liquidity + gauge operations |

---

## ✅ **Coverage Verification**

Each protocol's unique features are thoroughly tested:

- ✅ **Balancer**: Vault architecture, pool types, BPT handling
- ✅ **Uniswap V3**: Concentrated liquidity, NFT positions, tick math
- ✅ **Velodrome**: Gauge staking, CL pools, dual rewards

The test suite ensures that each protocol's specific mechanics are properly validated while maintaining a consistent testing framework across all protocols.
