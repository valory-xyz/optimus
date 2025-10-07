# Protocol Tests Documentation

## Overview

This document provides comprehensive documentation for the integration tests in the Optimus DeFi protocol testing suite. The tests are designed to validate the functionality of various DeFi protocols including Balancer, Uniswap V3, Velodrome, and staking mechanisms.

## Test Structure

The integration tests are organized in the following structure:

```
tests/integration/
├── protocols/
│   ├── balancer/           # Balancer protocol tests
│   ├── uniswap_v3/         # Uniswap V3 protocol tests  
│   ├── velodrome/          # Velodrome protocol tests
│   ├── staking/            # Staking compliance tests
│   └── base/               # Base test infrastructure
├── fixtures/               # Test data fixtures
├── conftest.py            # Pytest configuration
├── run_integration_tests.py # Test runner script
└── pytest.ini            # Pytest settings
```

## Protocol Tests

### 1. Balancer Protocol Tests

**Location**: `tests/integration/protocols/balancer/`

#### Features Tested:
- **Pool Components** (`test_balancer_components.py`):
  - Mathematical calculations for proportional joins/exits
  - Join/Exit kind enum validation for different pool types (Weighted, Stable, ComposableStable)
  - Pool type detection logic
  - Pool share proportional math calculations
  - Amount adjustment for different token orders
  - Pool ID and token retrieval methods
  - Enter/exit method validation and execution
  - Query methods for proportional joins/exits

- **Contract Integration** (`test_balancer_contract_integration.py`):
  - Vault contract transaction encoding (join/exit pools)
  - Pool token queries and balance checks
  - Weighted pool contract interactions
  - MultiSend contract operations
  - Contract error handling and parameter validation
  - Gas estimation and deadline validation
  - Batch contract operations
  - Contract response parsing

- **End-to-End Workflows** (`test_balancer_e2e_workflows.py`):
  - Complete enter workflow (joining pools)
  - Complete exit workflow (leaving pools)
  - Proportional join/exit calculations
  - Amount adjustment workflows
  - Value update workflows

- **Transaction Generation** (`test_balancer_transaction_generation.py`):
  - Transaction encoding and validation
  - Slippage protection mechanisms
  - Multi-step transaction workflows

### 2. Uniswap V3 Protocol Tests

**Location**: `tests/integration/protocols/uniswap_v3/`

#### Features Tested:
- **Pool Components** (`test_uniswap_v3_components.py`):
  - Position management (mint, decrease liquidity, collect fees, burn)
  - Liquidity position queries
  - Pool token and fee retrieval
  - Tick spacing calculations
  - Tick range calculations for concentrated liquidity
  - Slippage protection for mint and decrease operations
  - MintParams validation

- **Contract Integration** (`test_uniswap_v3_contract_integration.py`):
  - Position Manager contract interactions (mint, decrease liquidity, collect, burn)
  - Pool contract queries (slot0, tokens, fee, tick spacing)
  - Position data retrieval
  - Transaction encoding and validation
  - Gas estimation for different transaction types
  - Deadline validation
  - Error handling and parameter validation
  - Contract address validation
  - Batch operations
  - Retry logic for failed calls

- **End-to-End Workflows** (`test_uniswap_v3_e2e_workflows.py`):
  - Complete position creation workflow
  - Position management workflows
  - Fee collection workflows
  - Liquidity adjustment workflows

- **Transaction Generation** (`test_uniswap_v3_transaction_generation.py`):
  - Transaction encoding for all position operations
  - Slippage protection calculations
  - Multi-step transaction sequences

### 3. Velodrome Protocol Tests

**Location**: `tests/integration/protocols/velodrome/`

#### Features Tested:
- **Pool Components** (`test_velodrome_components.py`):
  - Abstract class validation (VelodromePoolBehaviour cannot be instantiated)
  - Method signature validation
  - Expected method existence checks
  - Missing implementation detection

- **Contract Integration** (`test_velodrome_contract_integration.py`):
  - Pool contract interactions
  - Gauge contract operations
  - Voter contract interactions
  - Transaction encoding and validation

- **End-to-End Workflows** (`test_velodrome_e2e_workflows.py`):
  - Pool entry/exit workflows
  - Staking/unstaking LP tokens
  - Reward claiming workflows
  - Gauge interactions

- **Transaction Generation** (`test_velodrome_transaction_generation.py`):
  - Transaction encoding for pool operations
  - Staking transaction generation
  - Reward claiming transactions

### 4. Staking Compliance Tests

**Location**: `tests/integration/protocols/staking/`

#### Features Tested:
- **Staking Integration** (`test_staking_integration.py`):
  - Complete staking compliance workflows
  - Checkpoint execution workflows
  - Vanity transaction workflows
  - KPI monitoring workflows
  - Compliance enforcement workflows
  - Service eviction workflows
  - Error handling workflows
  - Chain handling workflows
  - Timing workflows
  - Transaction counting workflows
  - End-to-end compliance workflows

- **Staking Compliance** (`test_staking_compliance.py`):
  - KPI compliance workflows
  - Checkpoint execution workflows
  - Vanity transaction workflows
  - KPI monitoring workflows
  - Staking state management workflows
  - Compliance enforcement workflows
  - Service eviction workflows
  - Checkpoint timing workflows
  - Transaction counting workflows
  - Error handling workflows
  - End-to-end compliance workflows

- **Checkpoint Mechanisms** (`test_checkpoint_mechanisms.py`):
  - Checkpoint timing validation
  - Checkpoint preparation workflows
  - Service state management
  - Transaction requirement calculations

- **KPI Compliance** (`test_kpi_compliance.py`):
  - KPI calculation and validation
  - Transaction counting mechanisms
  - Compliance threshold checking
  - Vanity transaction generation

- **Vanity Transactions** (`test_vanity_transactions.py`):
  - Vanity transaction generation
  - Hash manipulation for compliance
  - Transaction payload processing

## Test Infrastructure

### Base Test Framework

**Location**: `tests/integration/protocols/base/`

#### Components:
- **ProtocolIntegrationTestBase** (`protocol_test_base.py`):
  - Common test setup and teardown
  - Mock ledger API setup
  - Mock contract instances
  - Test data generation
  - Mock response handling
  - Transaction validation utilities
  - Mock behaviour creation
  - Generator consumption helpers

- **Test Helpers** (`test_helpers.py`):
  - Test data generators
  - Assertion utilities
  - Mock response generators
  - Test scenario builders

- **Mock Contracts** (`mock_contracts.py`):
  - Mock contract instances for all protocols
  - Contract method mocking
  - Response data generation

### Fixtures

**Location**: `tests/integration/fixtures/`

#### Components:
- **Pool Data Fixtures** (`pool_data_fixtures.py`):
  - Balancer pool data (weighted, stable, 3-token pools)
  - Uniswap V3 pool data (different fee tiers)
  - Velodrome pool data (stable, volatile, CL pools)
  - Token address fixtures
  - User address fixtures
  - Amount fixtures
  - Position data fixtures
  - User asset fixtures
  - Price data fixtures
  - Mock HTTP responses

- **Contract Fixtures** (`contract_fixtures.py`):
  - Mock ledger API
  - Contract instance mocks
  - Response data mocks

- **Mock Responses** (`mock_responses.py`):
  - API response mocks
  - Contract response mocks
  - Error scenario mocks

## Test Categories

### 1. Component Tests
- **Purpose**: Test individual protocol components in isolation
- **Scope**: Mathematical calculations, enum validations, method signatures
- **Files**: `test_*_components.py`

### 2. Contract Integration Tests
- **Purpose**: Test contract interactions with mocked blockchain
- **Scope**: Transaction encoding, contract queries, error handling
- **Files**: `test_*_contract_integration.py`

### 3. End-to-End Workflow Tests
- **Purpose**: Test complete user workflows
- **Scope**: Full protocol interactions, multi-step processes
- **Files**: `test_*_e2e_workflows.py`

### 4. Transaction Generation Tests
- **Purpose**: Test transaction creation and validation
- **Scope**: Transaction encoding, slippage protection, gas estimation
- **Files**: `test_*_transaction_generation.py`

### 5. Staking Compliance Tests
- **Purpose**: Test staking and compliance mechanisms
- **Scope**: KPI monitoring, checkpoint execution, vanity transactions
- **Files**: `test_staking_*.py`

## Key Features Being Tested

### Mathematical Operations
- Proportional join/exit calculations
- Slippage protection calculations
- Tick range calculations (Uniswap V3)
- Pool share calculations (Balancer)

### Contract Interactions
- Transaction encoding and decoding
- Contract method calls
- Error handling and retry logic
- Gas estimation
- Parameter validation

### Protocol-Specific Features
- **Balancer**: Pool types, join/exit kinds, vault operations
- **Uniswap V3**: Concentrated liquidity, position management, fee collection
- **Velodrome**: Staking mechanisms, gauge interactions, reward claiming
- **Staking**: KPI compliance, checkpoint mechanisms, vanity transactions

### Workflow Validation
- Complete user journeys
- Multi-step transaction sequences
- Error recovery mechanisms
- State management

## Running Tests

### Test Runner
Use the provided test runner script:
```bash
python tests/integration/run_integration_tests.py --all
```

### Specific Protocol Tests
```bash
python tests/integration/run_integration_tests.py --protocol balancer
python tests/integration/run_integration_tests.py --protocol uniswap_v3
python tests/integration/run_integration_tests.py --protocol velodrome
```

### Specific Test Types
```bash
python tests/integration/run_integration_tests.py --test-type unit
python tests/integration/run_integration_tests.py --test-type contract
python tests/integration/run_integration_tests.py --test-type e2e
```

