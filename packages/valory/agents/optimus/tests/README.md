# Optimus Agent E2E Tests

This directory contains comprehensive End-to-End (E2E) tests for the Optimus autonomous trading agent. These tests validate the complete agent workflow from initialization through strategy execution.

## Overview

The Optimus agent is an autonomous liquidity trading system that:
- Monitors Safe wallet balances and transfers
- Discovers liquidity opportunities across multiple DEXs (Balancer, Velodrome, Uniswap)
- Evaluates strategies using risk metrics (Sharpe ratio, depth score, IL risk)
- Executes trades through Safe multisig transactions

## Test Architecture

### Core Components

1. **Agent Under Test**: Full Optimus agent with all skills and connections
2. **Mock Infrastructure**: 
   - Hardhat local blockchain with deployed contracts
   - JSON server mocking external APIs (Safe API, CoinGecko, Balancer subgraph)
   - Tendermint consensus for ABCI rounds
   - IPFS daemon for metadata storage

3. **Test Scenarios**: Multiple test classes covering different agent behaviors

## Test Files Structure

```
tests/
├── README.md                          # This file
├── test_optimus_e2e.py                # Main E2E test definitions
├── helpers/
│   ├── constants.py                   # Test configuration constants
│   ├── docker.py                      # Docker image definitions
│   ├── fixtures.py                    # Pytest fixtures for infrastructure
│   ├── contracts/                     # Smart contract mocks
│   │   ├── deploy_contracts.js        # Hardhat deployment script
│   │   ├── hardhat.config.js          # Hardhat configuration
│   │   ├── Dockerfile                 # Hardhat container image
│   │   └── mocks/                     # Mock contract implementations
│   └── data/
│       ├── json_server/
│       │   └── data.json              # Mock API responses
│       ├── balancer_subgraph.json     # Balancer pool data
│       └── coingecko_prices.json      # Token price data
```

## Test Scenarios

### 1. Happy Path Test (`TestEnd2EndOptimusSingleAgent`)

**What it tests**: Complete successful trading workflow

**Expected Flow**:
1. **FetchStrategiesRound**: Agent discovers available liquidity strategies
2. **GetPositionsRound**: Agent fetches current Safe balances and positions
3. **APRPopulationRound**: Agent calculates APRs for discovered opportunities
4. **EvaluateStrategyRound**: Agent evaluates strategies using risk metrics
5. **DecisionMakingRound**: Agent decides on optimal strategy
6. **PostTxSettlementRound**: Agent executes Safe transaction

**Success Criteria**:
- All rounds complete successfully
- Agent finds opportunities using balancer_pools_search strategy
- Safe transaction is prepared and executed
- Logs show "Transaction executed successfully"

### 2. Investment Cap Test (`TestEnd2EndOptimusInvestmentCap`)

**What it tests**: Agent behavior when investment threshold is reached

**Expected Flow**:
1. Agent discovers strategies normally
2. Agent detects investment cap reached ($950+ threshold)
3. Agent skips transaction preparation
4. No PostTxSettlementRound occurs

**Success Criteria**:
- Agent logs "Investment threshold reached, limiting actions"
- Agent logs "No actions to prepare"
- Test completes without transaction execution

### 3. TiP Blocking Test (`TestEnd2EndOptimusTiPBlocking`)

**What it tests**: Time-in-Position constraints preventing exits

**Expected Flow**:
1. Agent discovers current positions
2. Agent detects all positions are blocked by TiP requirements
3. Agent cannot exit positions due to time constraints
4. No transaction is prepared

**Success Criteria**:
- Agent logs "All positions blocked by TiP conditions"
- Agent logs "TiP blocking exit"
- Agent logs "No actions to prepare"

### 4. Withdrawal Mode Test (`TestEnd2EndOptimusWithdrawalMode`)

**What it tests**: Agent behavior during withdrawal requests

**Expected Flow**:
1. Agent detects withdrawal mode is enabled
2. Agent prioritizes exiting all positions
3. Agent pauses new investments

**Success Criteria**:
- Agent logs "Investing paused due to withdrawal request"
- Agent focuses on position exits only

## Mock Data Configuration

### Safe API Mock Data
- **Balances**: ETH (5.0), USDC (10.0), USDT (10.0)
- **Transfers**: 2 incoming ERC20 transfers (USDC and USDT)
- **Addresses**: Uses Hardhat deployment addresses

### Balancer Mock Data
- **Pool**: USDC/USDT weighted pool
- **TVL**: $2.5M
- **APR**: 25%
- **Risk Metrics**: Sharpe ratio 1.5, depth score 150.0

### CoinGecko Mock Data
- **ETH**: $4000
- **USDC/USDT**: $1.00 each

