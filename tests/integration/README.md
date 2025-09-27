# DeFi Protocol Integration Tests

This directory contains comprehensive integration tests for DeFi protocol interactions in the Optimus agent. The tests ensure proper integration, correct pool interactions, and accurate yield calculations across all supported protocols.

## Overview

The integration tests are organized into five main categories:

1. **Unit Integration Tests**: Test individual protocol components in isolation
2. **Contract Integration Tests**: Test contract interactions with mocked blockchain
3. **End-to-End Integration Tests**: Test complete protocol workflows
4. **Yield Calculation Tests**: Test accurate yield and APR calculations
5. **Transaction Generation Tests**: Test proper transaction encoding and parameters

## Supported Protocols

- **Balancer**: Weighted pools, stable pools, composable stable pools
- **Uniswap V3**: Concentrated liquidity, NFT position management, fee collection
- **Velodrome**: Stable/volatile AMM pairs, gauge staking, reward claiming

## Test Structure

```
tests/integration/
├── protocols/
│   ├── base/                    # Base test classes and utilities
│   │   ├── protocol_test_base.py
│   │   ├── mock_contracts.py
│   │   └── test_helpers.py
│   ├── balancer/                # Balancer protocol tests
│   │   ├── test_balancer_components.py
│   │   ├── test_balancer_contract_integration.py
│   │   ├── test_balancer_e2e_workflows.py
│   │   ├── test_balancer_yield_calculations.py
│   │   └── test_balancer_transaction_generation.py
│   ├── uniswap_v3/              # Uniswap V3 protocol tests
│   │   ├── test_uniswap_v3_components.py
│   │   ├── test_uniswap_v3_contract_integration.py
│   │   ├── test_uniswap_v3_e2e_workflows.py
│   │   ├── test_uniswap_v3_yield_calculations.py
│   │   └── test_uniswap_v3_transaction_generation.py
│   └── velodrome/               # Velodrome protocol tests
│       ├── test_velodrome_components.py
│       ├── test_velodrome_contract_integration.py
│       ├── test_velodrome_e2e_workflows.py
│       ├── test_velodrome_yield_calculations.py
│       └── test_velodrome_transaction_generation.py
├── fixtures/                    # Test fixtures and data
│   ├── contract_fixtures.py
│   ├── pool_data_fixtures.py
│   └── mock_responses.py
├── conftest.py                  # Pytest configuration
├── run_integration_tests.py     # Test runner script
└── README.md                    # This file
```

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist
```

2. Ensure the project is properly set up with all dependencies.

### Basic Usage

Run all integration tests:
```bash
python tests/integration/run_integration_tests.py --all
```

Run tests for a specific protocol:
```bash
python tests/integration/run_integration_tests.py --protocol balancer
python tests/integration/run_integration_tests.py --protocol uniswap_v3
python tests/integration/run_integration_tests.py --protocol velodrome
```

Run specific test types:
```bash
python tests/integration/run_integration_tests.py --test-type unit
python tests/integration/run_integration_tests.py --test-type contract
python tests/integration/run_integration_tests.py --test-type e2e
python tests/integration/run_integration_tests.py --test-type yield
python tests/integration/run_integration_tests.py --test-type transaction
```

### Advanced Options

Run tests with coverage:
```bash
python tests/integration/run_integration_tests.py --all --coverage
```

Run tests in parallel:
```bash
python tests/integration/run_integration_tests.py --all --parallel
```

Run tests with verbose output:
```bash
python tests/integration/run_integration_tests.py --all --verbose
```

### Using pytest directly

You can also run tests directly with pytest:

```bash
# Run all integration tests
pytest tests/integration/

# Run specific protocol tests
pytest tests/integration/protocols/balancer/
pytest tests/integration/protocols/uniswap_v3/
pytest tests/integration/protocols/velodrome/

# Run specific test types
pytest tests/integration/ -m unit
pytest tests/integration/ -m contract
pytest tests/integration/ -m e2e
pytest tests/integration/ -m yield
pytest tests/integration/ -m transaction

# Run with coverage
pytest tests/integration/ --cov=packages.valory.skills.liquidity_trader_abci --cov-report=html
```

## Test Categories

### 1. Unit Integration Tests

Test individual protocol components in isolation:
- Math calculations (tick math, proportional math, etc.)
- Parameter validation
- Business logic functions
- Data structure validation

**Example**: `test_balancer_math_proportional_calculations`

### 2. Contract Integration Tests

Test contract interactions with mocked blockchain:
- Transaction encoding/decoding
- Contract method calls
- Parameter validation
- Error handling

**Example**: `test_vault_join_pool_transaction_encoding`

### 3. End-to-End Integration Tests

Test complete protocol workflows:
- Pool selection and analysis
- Liquidity provision/withdrawal
- Position management
- Rebalancing strategies
- Emergency procedures

**Example**: `test_complete_balancer_liquidity_provision_workflow`

### 4. Yield Calculation Tests

Test accurate yield and APR calculations:
- Trading fees APR
- Gauge rewards APR
- Impermanent loss calculations
- Compound interest calculations
- Total APR calculations

**Example**: `test_balancer_apr_calculation_accuracy`

### 5. Transaction Generation Tests

Test proper transaction encoding and parameters:
- Transaction structure validation
- Gas estimation
- Deadline validation
- Parameter encoding
- Error handling

**Example**: `test_join_pool_transaction_parameter_validation`

## Test Data and Fixtures

The tests use comprehensive fixtures for:
- **Contract Mocks**: Realistic mock implementations of all protocol contracts
- **Pool Data**: Test data for different pool types and configurations
- **User Assets**: Mock user asset balances and positions
- **Price Data**: Token price data for calculations
- **Mock Responses**: HTTP API responses and blockchain data

## Mock Infrastructure

The tests use a sophisticated mocking system:
- **MockContractFactory**: Generates protocol-specific contract mocks
- **ProtocolIntegrationTestBase**: Base class with common test utilities
- **TestDataGenerator**: Creates realistic test data
- **TestAssertions**: Custom assertions for protocol-specific validation

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Realistic Data**: Use realistic test data that matches production scenarios
3. **Comprehensive Coverage**: Test both happy path and edge cases
4. **Clear Assertions**: Use descriptive assertions that clearly indicate what's being tested
5. **Mock Appropriately**: Mock external dependencies but test real business logic

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the project is properly set up
2. **Mock Failures**: Check that mock responses match expected contract interfaces
3. **Test Timeouts**: Some tests may take longer due to complex calculations
4. **Coverage Issues**: Ensure all code paths are covered by tests

### Debug Mode

Run tests with debug output:
```bash
pytest tests/integration/ -v -s --tb=long
```

### Test Specific Files

Run a specific test file:
```bash
pytest tests/integration/protocols/balancer/test_balancer_components.py -v
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers for test categorization
3. Use the provided fixtures and base classes
4. Ensure tests are deterministic and don't rely on external state
5. Add documentation for complex test scenarios

## Performance

The integration tests are designed to run efficiently:
- Parallel execution support
- Optimized mock responses
- Minimal external dependencies
- Fast test data generation

For large test suites, consider using parallel execution:
```bash
pytest tests/integration/ -n auto
```
