# Optimus/Modius Quick Reference Guide

## Essential Commands

### Setup & Deployment
```bash
# Environment setup
poetry install && poetry shell
autonomy init --reset --author valory --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
autonomy packages sync --update-packages

# Service deployment
autonomy fetch --local --service valory/optimus && cd optimus
autonomy build-image
cp path/to/keys.json .
autonomy deploy build --n 1 -ltm
autonomy deploy run --build-dir abci_build/
```

### Environment Variables (Essential)
```bash
export ETHEREUM_LEDGER_RPC=<your_ethereum_rpc>
export OPTIMISM_LEDGER_RPC=<your_optimism_rpc>
export BASE_LEDGER_RPC=<your_base_rpc>
export MODE_LEDGER_RPC=<your_mode_rpc>
export ALL_PARTICIPANTS='["<your_agent_address>"]'
export SAFE_CONTRACT_ADDRESSES='{"ethereum":"<eth_safe>","optimism":"<opt_safe>","base":"<base_safe>","mode":"<mode_safe>"}'
export TENDERLY_ACCESS_KEY=<your_tenderly_key>
export COINGECKO_API_KEY=<your_coingecko_key>
```

## System Overview

### Supported Networks
- **Ethereum**: Staking and governance
- **Optimism**: Primary trading (Uniswap V3, Balancer, Velodrome)
- **Base**: Secondary trading (Uniswap V3, Balancer)
- **Mode**: Emerging opportunities (Velodrome)

### Key Thresholds
- **Minimum APR**: 5% for first investment
- **Balanced Trading**: 33.74% improvement required to switch
- **Risky Trading**: 28.92% improvement required to switch
- **Portfolio Limits**: 40% per DEX, 60% per chain, 30% per asset

## FSM States Quick Reference

### Core Flow
```
FetchStrategies → CallCheckpoint → CheckStakingKPI → GetPositions → APRPopulation → EvaluateStrategy → DecisionMaking → PostTxSettlement
```

### Key Events
- `DONE`: Successful completion, move to next state
- `WAIT`: Pause execution, finish round
- `SETTLE`: Prepare transaction for execution
- `ERROR`: Handle error condition
- `NO_MAJORITY`: Retry consensus

## File Locations

### Configuration Files
- Agent config: `packages/valory/agents/optimus/aea-config.yaml`
- Service config: `packages/valory/services/optimus/service.yaml`
- FSM specs: `packages/valory/skills/*/fsm_specification.yaml`

### Data Files
- Portfolio: `data/portfolio.json`
- Positions: `data/current_positions.json`
- Funding events: `data/funding_events.json`

### Key Behaviors
- Fetch strategies: `packages/valory/skills/liquidity_trader_abci/behaviours/fetch_strategies.py`
- APR population: `packages/valory/skills/liquidity_trader_abci/behaviours/apr_population.py`
- Strategy evaluation: `packages/valory/skills/liquidity_trader_abci/behaviours/evaluate_strategy.py`

## Troubleshooting Quick Fixes

### Common Issues
```bash
# Check balances
cast balance $SAFE_ADDRESS --rpc-url $OPTIMISM_LEDGER_RPC

# Test API connectivity
curl -H "x-cg-api-key: $COINGECKO_API_KEY" "https://api.coingecko.com/api/v3/ping"

# Validate Safe
cast call $SAFE_ADDRESS "nonce()" --rpc-url $OPTIMISM_LEDGER_RPC

# Debug mode
autonomy deploy run --build-dir abci_build/ --debug
```

### Log Locations
- Agent logs: `/logs/agent.log`
- Skill logs: `/logs/liquidity_trader_abci.log`
- Transaction logs: `/logs/transactions.log`

## API Rate Limits
- **CoinGecko**: 20 calls/minute (with API key)
- **Merkl**: No documented limits
- **Blockchain RPCs**: Varies by provider

## Key Metrics to Monitor
- Portfolio value and ROI
- Transaction success rate
- APR accuracy
- Gas costs
- Epoch completion time

## Emergency Procedures
1. **High gas costs**: Pause operations, optimize timing
2. **API failures**: Check connectivity, verify keys
3. **Position failures**: Review contract addresses, check balances
4. **Consensus issues**: Restart service, check network connectivity

## Contract Addresses (Key)

### Optimism
- Uniswap V3 Position Manager: `0xC36442b4a4522E871399CD717aBDD847Ab11FE88`
- Balancer Vault: `0xBA12222222228d8Ba445958a75a0704d566BF2C8`
- USDC: `0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85`

### Base
- Uniswap V3 Position Manager: `0x03a520b32C04BF3bEEf7BF5d56E39E92d51752dd`
- Balancer Vault: `0xBA12222222228d8Ba445958a75a0704d566BF2C8`
- USDC: `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`

### Mode
- USDC: `0xd988097fb8612cc24eec14542bc03424c656005f`
- WETH: `0x4200000000000000000000000000000000000006`

## Performance Benchmarks
- **Epoch time**: < 60 minutes
- **Transaction success**: > 95%
- **API response time**: < 500ms
- **Portfolio update**: Every 6 hours
- **APR accuracy**: > 90%

## Security Checklist
- [ ] Private keys secured
- [ ] Safe addresses verified
- [ ] API keys rotated regularly
- [ ] RPC endpoints trusted
- [ ] Transaction simulation enabled
- [ ] Gas limits configured
- [ ] Slippage protection active

This quick reference provides immediate access to the most commonly needed information for operating and troubleshooting Optimus/Modius.
