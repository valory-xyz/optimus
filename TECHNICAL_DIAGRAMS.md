# Optimus/Modius Technical Diagrams

This document contains detailed technical diagrams and flowcharts to complement the master documentation.

## System Architecture Overview

```mermaid
graph TB
    subgraph "External APIs"
        CG[CoinGecko API]
        MK[Merkl Platform]
        TD[Tenderly]
        BE[Blockchain Explorers]
    end
    
    subgraph "Agent Service Layer"
        AS[Agent Service]
        CS[Consensus Engine]
        SM[State Machine]
    end
    
    subgraph "Core Skills"
        LT[Liquidity Trader ABCI]
        OA[Optimus ABCI]
    end
    
    subgraph "Behaviors"
        FS[Fetch Strategies]
        AP[APR Population]
        ES[Evaluate Strategy]
        DM[Decision Making]
        TS[Tx Settlement]
    end
    
    subgraph "Protocol Integrations"
        UP[Uniswap Pools]
        BP[Balancer Pools]
        VP[Velodrome Pools]
        SP[Sturdy Pools]
    end
    
    subgraph "Data Storage"
        KV[KV Store]
        PD[Portfolio Data]
        FE[Funding Events]
        GC[Gas Costs]
    end
    
    subgraph "Blockchain Networks"
        ETH[Ethereum]
        OPT[Optimism]
        BASE[Base]
        MODE[Mode]
    end
    
    CG --> FS
    MK --> FS
    TD --> DM
    BE --> FS
    
    AS --> CS
    CS --> SM
    SM --> LT
    SM --> OA
    
    LT --> FS
    LT --> AP
    LT --> ES
    LT --> DM
    LT --> TS
    
    FS --> UP
    FS --> BP
    FS --> VP
    FS --> SP
    
    FS --> KV
    AP --> KV
    ES --> PD
    DM --> FE
    TS --> GC
    
    DM --> ETH
    DM --> OPT
    DM --> BASE
    DM --> MODE
```

## Finite State Machine Flow

### Liquidity Trader ABCI FSM

```mermaid
stateDiagram-v2
    [*] --> FetchStrategiesRound
    
    FetchStrategiesRound --> CallCheckpointRound : DONE
    FetchStrategiesRound --> FetchStrategiesRound : WAIT/NO_MAJORITY/ROUND_TIMEOUT
    
    CallCheckpointRound --> CheckStakingKPIMetRound : DONE/NEXT_CHECKPOINT_NOT_REACHED_YET
    CallCheckpointRound --> GetPositionsRound : SERVICE_NOT_STAKED/SERVICE_EVICTED
    CallCheckpointRound --> FinishedCallCheckpointRound : SETTLE
    CallCheckpointRound --> CallCheckpointRound : NO_MAJORITY/ROUND_TIMEOUT
    
    CheckStakingKPIMetRound --> GetPositionsRound : DONE/STAKING_KPI_MET/STAKING_KPI_NOT_MET/ERROR
    CheckStakingKPIMetRound --> FinishedCheckStakingKPIMetRound : SETTLE
    CheckStakingKPIMetRound --> CheckStakingKPIMetRound : NO_MAJORITY/ROUND_TIMEOUT
    
    GetPositionsRound --> APRPopulationRound : DONE
    GetPositionsRound --> GetPositionsRound : NO_MAJORITY/ROUND_TIMEOUT
    
    APRPopulationRound --> EvaluateStrategyRound : DONE
    APRPopulationRound --> APRPopulationRound : NONE/NO_MAJORITY/ROUND_TIMEOUT
    
    EvaluateStrategyRound --> DecisionMakingRound : DONE
    EvaluateStrategyRound --> FinishedEvaluateStrategyRound : WAIT
    EvaluateStrategyRound --> EvaluateStrategyRound : NO_MAJORITY/ROUND_TIMEOUT
    
    DecisionMakingRound --> FinishedDecisionMakingRound : DONE/ERROR
    DecisionMakingRound --> FinishedTxPreparationRound : SETTLE
    DecisionMakingRound --> DecisionMakingRound : UPDATE/NO_MAJORITY/ROUND_TIMEOUT
    
    FinishedDecisionMakingRound --> PostTxSettlementRound
    FinishedTxPreparationRound --> PostTxSettlementRound
    
    PostTxSettlementRound --> DecisionMakingRound : ACTION_EXECUTED
    PostTxSettlementRound --> CallCheckpointRound : CHECKPOINT_TX_EXECUTED
    PostTxSettlementRound --> CheckStakingKPIMetRound : VANITY_TX_EXECUTED
    PostTxSettlementRound --> FailedMultiplexerRound : UNRECOGNIZED
    PostTxSettlementRound --> PostTxSettlementRound : DONE/NO_MAJORITY/ROUND_TIMEOUT
    
    FinishedCallCheckpointRound --> [*]
    FinishedCheckStakingKPIMetRound --> [*]
    FinishedEvaluateStrategyRound --> [*]
    FinishedDecisionMakingRound --> [*]
    FinishedTxPreparationRound --> [*]
    FailedMultiplexerRound --> [*]
```

### Optimus ABCI Extended FSM

```mermaid
stateDiagram-v2
    [*] --> RegistrationStartupRound
    
    RegistrationStartupRound --> FetchStrategiesRound : DONE
    RegistrationRound --> FetchStrategiesRound : DONE
    RegistrationRound --> RegistrationRound : NO_MAJORITY
    
    FetchStrategiesRound --> CallCheckpointRound : DONE
    
    CallCheckpointRound --> RandomnessTransactionSubmissionRound : SETTLE
    CheckStakingKPIMetRound --> RandomnessTransactionSubmissionRound : SETTLE
    DecisionMakingRound --> RandomnessTransactionSubmissionRound : SETTLE
    ResetRound --> RandomnessTransactionSubmissionRound : DONE
    
    RandomnessTransactionSubmissionRound --> SelectKeeperTransactionSubmissionARound : DONE
    RandomnessTransactionSubmissionRound --> RandomnessTransactionSubmissionRound : NO_MAJORITY/ROUND_TIMEOUT
    
    SelectKeeperTransactionSubmissionARound --> CollectSignatureRound : DONE
    SelectKeeperTransactionSubmissionARound --> ResetAndPauseRound : INCORRECT_SERIALIZATION
    SelectKeeperTransactionSubmissionARound --> ResetRound : NO_MAJORITY
    
    CollectSignatureRound --> FinalizationRound : DONE
    CollectSignatureRound --> ResetRound : NO_MAJORITY
    
    FinalizationRound --> ValidateTransactionRound : DONE
    FinalizationRound --> SelectKeeperTransactionSubmissionBRound : FINALIZATION_FAILED/INSUFFICIENT_FUNDS
    FinalizationRound --> SelectKeeperTransactionSubmissionBAfterTimeoutRound : FINALIZE_TIMEOUT
    FinalizationRound --> CheckTransactionHistoryRound : CHECK_HISTORY
    FinalizationRound --> SynchronizeLateMessagesRound : CHECK_LATE_ARRIVING_MESSAGE
    
    ValidateTransactionRound --> PostTxSettlementRound : DONE
    ValidateTransactionRound --> CheckTransactionHistoryRound : NEGATIVE/VALIDATE_TIMEOUT
    ValidateTransactionRound --> SelectKeeperTransactionSubmissionBRound : NONE
    
    PostTxSettlementRound --> DecisionMakingRound : ACTION_EXECUTED
    PostTxSettlementRound --> CallCheckpointRound : CHECKPOINT_TX_EXECUTED
    PostTxSettlementRound --> CheckStakingKPIMetRound : VANITY_TX_EXECUTED
    PostTxSettlementRound --> ResetAndPauseRound : UNRECOGNIZED
    
    ResetAndPauseRound --> FetchStrategiesRound : DONE
    ResetAndPauseRound --> RegistrationRound : NO_MAJORITY/RESET_AND_PAUSE_TIMEOUT
```

## Data Flow Architecture

### Portfolio Calculation Flow

```mermaid
sequenceDiagram
    participant FS as FetchStrategies
    participant BC as Blockchain
    participant CG as CoinGecko
    participant PD as PortfolioData
    participant KV as KVStore
    
    FS->>BC: Get current positions
    BC-->>FS: Position data
    
    FS->>BC: Get token balances
    BC-->>FS: Balance data
    
    FS->>CG: Get token prices
    CG-->>FS: Price data
    
    FS->>FS: Calculate position values
    FS->>FS: Calculate safe balances
    FS->>FS: Aggregate portfolio
    
    FS->>PD: Store portfolio data
    FS->>KV: Cache calculations
```

### Strategy Evaluation Flow

```mermaid
sequenceDiagram
    participant AP as APRPopulation
    participant MK as Merkl
    participant ES as EvaluateStrategy
    participant DM as DecisionMaking
    participant KV as KVStore
    
    AP->>MK: Fetch campaigns
    MK-->>AP: Campaign data
    
    AP->>AP: Calculate APRs
    AP->>KV: Store APR data
    
    ES->>KV: Read current positions
    ES->>KV: Read APR data
    ES->>ES: Compare opportunities
    ES->>ES: Apply thresholds
    
    ES->>DM: Strategy recommendations
    DM->>DM: Make final decisions
    DM->>KV: Store decisions
```

### Transaction Execution Flow

```mermaid
sequenceDiagram
    participant DM as DecisionMaking
    participant TD as Tenderly
    participant BC as Blockchain
    participant TS as TxSettlement
    participant PD as PositionData
    
    DM->>DM: Prepare transaction
    DM->>TD: Simulate transaction
    TD-->>DM: Simulation result
    
    alt Simulation Success
        DM->>BC: Submit transaction
        BC-->>DM: Transaction hash
        
        DM->>TS: Monitor transaction
        TS->>BC: Check confirmation
        BC-->>TS: Confirmation status
        
        TS->>PD: Update positions
        TS->>PD: Record performance
    else Simulation Failed
        DM->>DM: Log error
        DM->>DM: Skip transaction
    end
```

## Protocol Integration Architecture

### DEX Integration Pattern

```mermaid
classDiagram
    class LiquidityTraderBaseBehaviour {
        +contract_interact()
        +_get_token_balance()
        +_get_token_decimals()
        +calculate_user_share_values()
    }
    
    class BalancerPoolBehaviour {
        +get_user_share_value_balancer()
        +_get_balancer_pool_name()
    }
    
    class UniswapPoolBehaviour {
        +get_user_share_value_uniswap()
        +_calculate_cl_position_value()
    }
    
    class VelodromePoolBehaviour {
        +get_user_share_value_velodrome()
        +_get_user_share_value_velodrome_cl()
        +_get_user_share_value_velodrome_non_cl()
    }
    
    LiquidityTraderBaseBehaviour <|-- BalancerPoolBehaviour
    LiquidityTraderBaseBehaviour <|-- UniswapPoolBehaviour
    LiquidityTraderBaseBehaviour <|-- VelodromePoolBehaviour
    
    class FetchStrategiesBehaviour {
        +async_act()
        +_handle_balancer_position()
        +_handle_uniswap_position()
        +_handle_velodrome_position()
        +_handle_sturdy_position()
    }
    
    BalancerPoolBehaviour <|-- FetchStrategiesBehaviour
    UniswapPoolBehaviour <|-- FetchStrategiesBehaviour
    VelodromePoolBehaviour <|-- FetchStrategiesBehaviour
```

### Position Management Data Model

```mermaid
erDiagram
    POSITION {
        string pool_address
        string pool_id
        string chain
        string dex_type
        string status
        string token0
        string token0_symbol
        string token1
        string token1_symbol
        bigint amount0
        bigint amount1
        bigint current_liquidity
        int token_id
        float apr
        int timestamp
        string tx_hash
        int enter_timestamp
        int exit_timestamp
    }
    
    PORTFOLIO {
        float portfolio_value
        float value_in_pools
        float value_in_safe
        float initial_investment
        float volume
        string agent_hash
        string address
        int last_updated
    }
    
    ALLOCATION {
        string chain
        string type
        string id
        array assets
        float apr
        string details
        float ratio
        string address
        array tick_ranges
    }
    
    PORTFOLIO_BREAKDOWN {
        string asset
        string address
        float balance
        float price
        float value_usd
        float ratio
    }
    
    FUNDING_EVENT {
        string from_address
        float amount
        string token_address
        string symbol
        string timestamp
        string tx_hash
        string type
    }
    
    PORTFOLIO ||--o{ ALLOCATION : contains
    PORTFOLIO ||--o{ PORTFOLIO_BREAKDOWN : contains
    POSITION ||--o{ ALLOCATION : generates
```

## Risk Management Framework

### Portfolio Diversification Limits

```mermaid
pie title Portfolio Allocation Limits
    "Single DEX Max" : 40
    "Single Chain Max" : 60
    "Single Asset Max" : 30
    "Available for New" : 70
```

### Decision Making Process

```mermaid
flowchart TD
    A[New Opportunity Detected] --> B{APR > 5%?}
    B -->|No| C[Reject Opportunity]
    B -->|Yes| D{Current Position Exists?}
    
    D -->|No| E[Calculate Position Size]
    D -->|Yes| F{Trading Type?}
    
    F -->|Balanced| G{New APR > Current APR * 1.3374?}
    F -->|Risky| H{New APR > Current APR * 1.2892?}
    
    G -->|No| I[Stay in Current Position]
    G -->|Yes| J[Plan Position Switch]
    H -->|No| I
    H -->|Yes| J
    
    E --> K{Portfolio Limits OK?}
    J --> K
    
    K -->|No| L[Reject - Risk Limits]
    K -->|Yes| M{Gas Cost Acceptable?}
    
    M -->|No| N[Reject - High Gas]
    M -->|Yes| O[Execute Transaction]
    
    O --> P[Update Positions]
    P --> Q[Record Performance]
```

## Multi-Chain Coordination

### Cross-Chain State Management

```mermaid
graph TB
    subgraph "Ethereum Mainnet"
        ES[Staking Contracts]
        EG[Governance]
    end
    
    subgraph "Optimism"
        OU[Uniswap V3]
        OB[Balancer]
        OV[Velodrome]
        OS[Safe Contract]
    end
    
    subgraph "Base"
        BU[Uniswap V3]
        BB[Balancer]
        BS[Safe Contract]
    end
    
    subgraph "Mode"
        MV[Velodrome]
        MS[Safe Contract]
        MU[USDC Bridge]
    end
    
    subgraph "Agent Service"
        AS[Consensus Engine]
        SM[State Manager]
        PM[Portfolio Manager]
    end
    
    AS --> ES
    AS --> OS
    AS --> BS
    AS --> MS
    
    SM --> OU
    SM --> OB
    SM --> OV
    SM --> BU
    SM --> BB
    SM --> MV
    
    PM --> OS
    PM --> BS
    PM --> MS
    
    OS -.->|Bridge| BS
    BS -.->|Bridge| MS
    MS -.->|Bridge| OS
```

### Asset Flow Diagram

```mermaid
sankey-beta
    USDC,Ethereum,1000
    USDC,Optimism,800
    USDC,Base,150
    USDC,Mode,50
    
    Optimism,Uniswap V3,400
    Optimism,Balancer,250
    Optimism,Velodrome,150
    
    Base,Uniswap V3,100
    Base,Balancer,50
    
    Mode,Velodrome,50
```

## Performance Monitoring Dashboard

### Key Metrics Visualization

```mermaid
graph LR
    subgraph "Portfolio Metrics"
        PV[Portfolio Value: $1,500]
        ROI[ROI 30d: 8.7%]
        SR[Sharpe Ratio: 1.8]
        MD[Max Drawdown: -3.2%]
    end
    
    subgraph "Operational Metrics"
        TSR[Tx Success: 95%]
        AGC[Avg Gas: $2.50]
        ECT[Epoch Time: 45min]
        ART[API Response: 250ms]
    end
    
    subgraph "Strategy Metrics"
        AA[APR Accuracy: 92%]
        PD[Position Duration: 7d]
        RF[Rebalance Freq: 2/week]
        RAR[Risk-Adj Return: 12%]
    end
    
    subgraph "Alert Status"
        CA[Critical: 0]
        WA[Warning: 2]
        IA[Info: 5]
    end
```

## Deployment Architecture

### Service Deployment Flow

```mermaid
flowchart TD
    A[Environment Setup] --> B[Install Dependencies]
    B --> C[Configure Environment Variables]
    C --> D[Deploy Safes on All Chains]
    D --> E[Fund Safes with ETH & USDC]
    E --> F[Generate Agent Keys]
    F --> G[Build Docker Image]
    G --> H[Deploy Service]
    H --> I[Health Checks]
    I --> J{All Checks Pass?}
    J -->|No| K[Debug Issues]
    J -->|Yes| L[Start Operations]
    K --> I
    L --> M[Monitor Performance]
```

### Container Architecture

```mermaid
graph TB
    subgraph "Docker Container"
        subgraph "Agent Service"
            AS[Autonomy Agent]
            CS[Consensus Service]
        end
        
        subgraph "Skills"
            LT[Liquidity Trader ABCI]
            OA[Optimus ABCI]
        end
        
        subgraph "Data Layer"
            KV[KV Store]
            FS[File System]
        end
        
        subgraph "Network Layer"
            HTTP[HTTP Client]
            WS[WebSocket Client]
            RPC[RPC Client]
        end
    end
    
    subgraph "External Services"
        BC[Blockchain Networks]
        API[External APIs]
        DB[External Database]
    end
    
    AS --> LT
    AS --> OA
    CS --> AS
    
    LT --> KV
    OA --> KV
    LT --> FS
    
    HTTP --> API
    RPC --> BC
    WS --> DB
    
    LT --> HTTP
    LT --> RPC
    OA --> WS
```

This technical diagrams document provides visual representations of the key architectural components, data flows, and operational processes described in the master documentation. These diagrams help visualize the complex interactions between different system components and provide a clearer understanding of how Optimus/Modius operates.
