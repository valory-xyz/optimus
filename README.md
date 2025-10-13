# Optimus Service

## Overview

Optimus is an autonomous agent service that automatically manages liquidity provision on the Optimism network. Built on the Open Autonomy framework, Optimus uses sophisticated finite state machines to coordinate actions across multiple agents without human intervention.

The system identifies, evaluates, and executes profitable liquidity provision opportunities by integrating with major decentralized exchanges including Balancer, Uniswap V3, and Velodrome. It automatically tracks performance metrics while providing real-time portfolio management with comprehensive analytics and risk assessment.

Optimus operates as a decentralized autonomous system that can handle complex multi-step operations including token swapping and liquidity provision. It maintains detailed records of all activities and continuously optimizes its strategies based on market conditions and performance data.

## Supported Protocols

Optimus integrates with the following decentralized exchanges and protocols:

### Balancer
- Supports both weighted pools and stable pools with different risk and return profiles
- Weighted pools allow for custom asset ratios and exposure to multiple tokens simultaneously
- Stable pools focus on correlated assets with lower impermanent loss risk
- Direct interaction with the Balancer Vault contract for maximum efficiency
- Handles complex pool compositions with multiple assets and custom weights

### Uniswap V3
- Specializes in concentrated liquidity positions with custom price ranges
- Supports multiple fee tiers to optimize for different trading patterns
- NFT-based position management through the Position Manager contract
- Allows multiple positions in the same pool with different ranges

### Velodrome
- Supports stable pools, volatile pools, and concentrated liquidity pools
- Advanced mathematical models to calculate optimized liquidity bands for Concentrated Pools
- Calculates different liquidity bands for optimal positioning
- Volatile pools use traditional constant product formulas
- Handles additional VELO rewards for liquidity providers

## Operational Process

### Happy Path - Typical Operation Cycle

A typical successful operation cycle for Optimus follows a predictable pattern that demonstrates how the system identifies and executes profitable trading opportunities:

1. **System Initialization & Registration**: All agents register with the network and establish their identities and permissions.

2. **Strategy Fetching**: The system scans multiple sources for available trading opportunities using GraphQL endpoints and various data sources. The agent maintains a whitelist (primarily stablecoins) and filters opportunities based on this whitelist. For each strategy, the system calculates three critical metrics:
   - **Sharpe ratio**: Measures risk-adjusted returns
   - **Depth score**: Evaluates liquidity depth and market stability
   - **Impermanent loss risk score**: Assesses potential losses from price divergence

3. **Portfolio State Retrieval**: The system fetches active liquidity positions from all integrated protocols, updates position statuses, and calculates current token balances and USD values for all holdings.

4. **Evaluation Phase**: All available opportunities are assessed against profitability and risk criteria. The three metrics are combined into a composite score that provides comprehensive evaluation. The system selects the best strategies based on these composite scores.

5. **Decision-Making**: The system determines final execution decisions and action order. It handles complex multi-step operations including swapping and liquidity provision while managing transaction routing and optimization to minimize costs and maximize efficiency.

6. **Transaction Submission**: Coordinates with the transaction settlement system to execute the prepared transactions.

7. **Post-Transaction Settlement**: Updates position records, processes transaction receipts, extracts relevant event data, and updates portfolio valuations and performance metrics.

8. **Cycle Repeat**: The system returns to the beginning of the cycle, continuously seeking to optimize its portfolio and capture the best available yields.

### Detailed Round Explanations

#### FetchStrategiesRound
The primary entry point for each trading cycle, responsible for gathering all information needed for informed trading decisions. This round:
- Retrieves available trading strategies/protocols selected by the user via Chat UI
- Performs comprehensive portfolio updates by recalculating current position values and performance metrics
- Filters whitelisted assets based on recent price changes (removes assets with >5% price drops in 24 hours)
- Manages funding events and tracks initial investment values
- Calculates user portfolio including initially invested funds, current holdings, and allocations
- Displays portfolio information in the agent profile for transparency
- Triggers portfolio recalculation based on time intervals (typically every 2 hours) or significant position changes

#### CallCheckpointRound
Manages the critical function of maintaining eligibility for Olas Staking Rewards through proper checkpoint management. This round:
- Checks whether checkpoint conditions have been met based on predefined time intervals and transaction counts
- Validates the current service staking status
- Calculates minimum safe transactions required to maintain staking eligibility
- Compares actual transactions executed against requirements to determine KPI threshold compliance
- Prepares and submits checkpoint transactions to the staking contract when conditions are satisfied
- Handles edge cases such as service not staked, checkpoint timestamps not reached, or service eviction scenarios
- Maintains detailed logging of checkpoint status, transaction counts, and timing information

#### CheckStakingKPIMetRound
Verifies whether Key Performance Indicators for staking rewards have been satisfied and helps the agent reach its KPI if not yet achieved. This round:
- Calculates required number of transactions within the current epoch based on liveness ratio and period parameters
- Compares actual transaction volume against requirements to determine compliance status
- Determines reward eligibility based on comprehensive KPI analysis
- Initiates vanity transactions (zero-value transactions) if KPIs are not met after a certain time to ensure agents reach KPI and earn staking rewards

#### GetPositionsRound
Retrieves and updates comprehensive information about the system's current portfolio state across all supported networks and protocols. This round:
- Fetches active liquidity positions from all integrated protocols (Balancer, Uniswap V3, Velodrome)
- Queries each protocol's contracts for current position data including liquidity amounts, token balances, and position status
- Updates position statuses by checking for closed, liquidated, or modified positions
- Identifies positions with zero liquidity and marks them as closed
- Calculates current token balances and USD values for all holdings
- Queries token contracts for precise balance information and applies current market prices

#### APRPopulationRound
Calculates accurate APRs for the agent's performance, providing critical profitability data for dashboard displays. This round:
- Calculates the APR and ETH-adjusted APR of the agent
- Creates a portfolio snapshot stored in the centralized database
- Provides data for displaying average performance of agents on dashboards

#### EvaluateStrategyRound
Performs comprehensive analysis of all available strategies to determine which opportunities merit execution. This round:
- Fetches trading opportunities from multiple DEXs/protocols (Balancer, Uniswap, Velodrome)
- Performs risk factor assessment including impermanent loss potential, historical performance, and liquidity depth analysis
- Determines optimal allocation amounts based on portfolio size, risk tolerance, and diversification requirements
- Calculates appropriate position sizes that maximize returns while maintaining acceptable risk levels
- Ranks all available opportunities based on profitability, risk, compatibility, and strategic objectives
- Creates action orders based on strategy selection and current agent state (e.g., swapping for required assets, exiting previous positions)

#### DecisionMakingRound
Makes final execution decisions and prepares detailed transaction data. This round:
- Handles complex multi-step operations including token swapping and liquidity provision
- Coordinates operations to ensure efficient execution with minimal slippage and costs
- Performs transaction routing and optimization to identify the most efficient paths
- Prepares different types of actions (entering new pools, exiting existing positions, claiming rewards)
- Validates transaction feasibility by checking balances, estimating gas costs, and simulating execution
- Performs risk management checks to ensure proposed actions align with portfolio risk parameters
- Maintains diversification requirements while pursuing attractive opportunities
- Coordinates with the transaction settlement system for proper handoff

#### PostTxSettlementRound
Handles all post-transaction processing and state updates to ensure the system accurately reflects transaction results. This round:
- Updates position records after successful transaction execution
- Processes transaction receipts and extracts relevant event data
- Identifies new positions created, existing positions modified, and positions closed
- Parses blockchain event logs to extract precise information about transaction outcomes
- Records exact token amounts received, liquidity tokens minted, and any additional rewards or fees

## Architecture

The Optimus service is an [agent service](https://docs.autonolas.network/open-autonomy/get_started/what_is_an_agent_service/) (or autonomous service) based on the [Open Autonomy framework](https://docs.autonolas.network/open-autonomy/).

## Setup Instructions

Below are the steps to prepare your environment, configure the agent keys, and run the service.

### Prepare the Environment

System requirements:

- Python `== 3.10`
- [Poetry](https://python-poetry.org/docs/) `>=1.4.0`
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

Clone this repository:

```bash
git clone https://github.com/valory-xyz/optimus.git
```

Create a development environment:

```bash
poetry install && poetry shell
```

Configure the Open Autonomy framework:

```bash
autonomy init --reset --author valory --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
```

Pull packages required to run the service:

```bash
autonomy packages sync --update-packages
```

### Prepare the Keys and the Safe

1. **Gnosis Keypair**: Prepare the `keys.json` file with the Gnosis keypair of your agent (replace the uppercase placeholders):

    ```bash
    cat > keys.json << EOF
    [
      {
        "address": "YOUR_AGENT_ADDRESS",
        "private_key": "YOUR_AGENT_PRIVATE_KEY"
      }
    ]
    EOF
    ```

2. **Deploy Safes**: You need to deploy [Safes](https://safe.global/) on the following networks:
   - Optimism

3. **Fund Accounts**: 
   - Provide ETH and USDC to your Safe address on Optimsim
   - Provide ETH to your agent on Optimism to cover gas fees

4. **Tenderly Credentials**: Obtain your Tenderly Access Key, Account Slug, and Project Slug from https://dashboard.tenderly.co/ under settings

5. **CoinGecko API Key**: Get an API key from https://www.coingecko.com/ under My Account → Developer's Dashboard

### Configure the Service

Set up the following environment variables:

```bashexport OPTIMISM_LEDGER_RPC=INSERT_YOUR_OPTIMISM_RPC

export ALL_PARTICIPANTS='["YOUR_AGENT_ADDRESS"]'
export SAFE_CONTRACT_ADDRESSES='{"optimism":"YOUR_SAFE_ADDRESS_ON_OPTIMISM"}'

export SLIPPAGE_FOR_SWAP=0.09
export TENDERLY_ACCESS_KEY=YOUR_TENDERLY_ACCESS_KEY
export TENDERLY_ACCOUNT_SLUG=YOUR_TENDERLY_ACCOUNT_SLUG
export TENDERLY_PROJECT_SLUG=YOUR_TENDERLY_PROJECT_SLUG
export COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY
```

**Note**: The default value for `SLIPPAGE_FOR_SWAP` is provided, but feel free to experiment with different values. It indicates the allowed slippage when bridging/swapping assets using LiFi.

### Run the Service

Once you have configured the environment variables:

1. Fetch the service:

    ```bash
    autonomy fetch --local --service valory/optimus && cd optimus
    ```

2. Build the Docker image:

    ```bash
    autonomy build-image
    ```

3. Copy your `keys.json` file to the current directory:

    ```bash
    cp path/to/keys.json .
    ```

4. Build the deployment with a single agent and run:

    ```bash
    autonomy deploy build --n 1 -ltm
    autonomy deploy run --build-dir abci_build/
    ```
