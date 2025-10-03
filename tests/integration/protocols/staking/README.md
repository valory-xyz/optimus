# Staking and Compliance Tests

This directory contains comprehensive integration tests for staking and compliance mechanisms in the Optimus service.

## 🎯 Test Overview

The staking compliance tests ensure that agents maintain proper OLAS staking requirements and execute necessary compliance transactions to continue earning rewards.

## 📁 Test Structure

```
tests/integration/protocols/staking/
├── __init__.py
├── fixtures/
│   ├── __init__.py
│   └── staking_fixtures.py          # Mock contracts and test data
├── test_staking_compliance.py      # Main compliance tests
├── test_checkpoint_mechanisms.py   # Checkpoint-specific tests
├── test_vanity_transactions.py     # Vanity transaction tests
├── test_kpi_compliance.py          # KPI monitoring tests
├── test_staking_integration.py     # End-to-end integration tests
└── README.md                       # This file
```

## 🚀 Running the Tests

### Method 1: Direct pytest execution

```bash
# Run all staking tests
pytest tests/integration/protocols/staking/ -v

# Run specific test file
pytest tests/integration/protocols/staking/test_staking_compliance.py -v

# Run specific test method
pytest tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_staking_kpi_compliance_workflow -v

# Run with coverage
pytest tests/integration/protocols/staking/ --cov=packages.valory.skills.liquidity_trader_abci --cov-report=html
```

### Method 2: Using tox (recommended)

```bash
# Run all tests including staking tests
tox -e py3.10-linux

# Run only staking tests with tox
tox -e py3.10-linux -- tests/integration/protocols/staking/

# Run protocol-specific tests with dedicated tox environments
tox -e test-protocol-staking      # Staking compliance tests
tox -e test-protocol-balancer     # Balancer protocol tests  
tox -e test-protocol-uniswap      # Uniswap V3 protocol tests
tox -e test-protocol-velodrome    # Velodrome protocol tests
```

### Method 3: Using the integration test runner

```bash
# Run all integration tests
python tests/integration/run_integration_tests.py --all

# Run staking tests specifically
python tests/integration/run_integration_tests.py --protocol staking
```

### Method 4: Individual test execution

```bash
# Run specific test categories
pytest tests/integration/protocols/staking/test_checkpoint_mechanisms.py -v
pytest tests/integration/protocols/staking/test_vanity_transactions.py -v
pytest tests/integration/protocols/staking/test_kpi_compliance.py -v
pytest tests/integration/protocols/staking/test_staking_integration.py -v
```

## 🧪 Test Categories

### 1. **Staking Compliance Tests** (`test_staking_compliance.py`)
- **KPI Compliance Workflow**: Tests normal KPI compliance flow
- **Checkpoint Execution**: Tests checkpoint transaction execution
- **Vanity Transaction Workflow**: Tests vanity transaction execution
- **KPI Monitoring**: Tests KPI requirement monitoring
- **Staking State Management**: Tests staking state transitions
- **Compliance Enforcement**: Tests compliance enforcement mechanisms
- **Service Eviction**: Tests service eviction scenarios
- **Error Handling**: Tests error handling and recovery

### 2. **Checkpoint Mechanisms** (`test_checkpoint_mechanisms.py`)
- **Timing Accuracy**: Tests checkpoint timing validation
- **Transaction Encoding**: Tests checkpoint transaction encoding
- **State Validation**: Tests staking state validation
- **Failure Recovery**: Tests checkpoint failure recovery
- **Edge Cases**: Tests timing boundary conditions
- **Chain Handling**: Tests multi-chain support
- **Contract Interaction**: Tests contract interaction
- **Error Handling**: Tests exception scenarios

### 3. **Vanity Transactions** (`test_vanity_transactions.py`)
- **Transaction Generation**: Tests vanity transaction creation
- **Execution Workflow**: Tests vanity transaction execution
- **Contract Interaction**: Tests safe contract interaction
- **Failure Handling**: Tests error scenarios
- **Hash Processing**: Tests transaction hash validation
- **Logging**: Tests comprehensive logging
- **Compliance Restoration**: Tests KPI compliance restoration
- **Parameter Validation**: Tests parameter validation

### 4. **KPI Compliance** (`test_kpi_compliance.py`)
- **Requirement Calculation**: Tests KPI requirement calculation
- **Transaction Counting**: Tests transaction counting accuracy
- **Threshold Evaluation**: Tests KPI threshold evaluation
- **Enforcement Triggers**: Tests compliance enforcement
- **Service Eviction**: Tests eviction scenarios
- **Error Handling**: Tests invalid data handling
- **Period Thresholds**: Tests time-based compliance
- **Chain Handling**: Tests multi-chain support

### 5. **Integration Tests** (`test_staking_integration.py`)
- **Complete Workflows**: Tests end-to-end staking compliance
- **Checkpoint Workflows**: Tests full checkpoint execution
- **Vanity Transaction Workflows**: Tests complete vanity tx flow
- **KPI Monitoring Workflows**: Tests comprehensive KPI monitoring
- **Compliance Enforcement**: Tests full compliance enforcement
- **Service Eviction**: Tests complete eviction workflows
- **Error Handling**: Tests comprehensive error handling
- **Chain Handling**: Tests multi-chain workflows
- **Timing Workflows**: Tests complete timing validation
- **Transaction Counting**: Tests full transaction counting
- **End-to-End**: Tests complete system integration

## 🔧 Test Configuration

### Environment Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Install test dependencies
pip install pytest pytest-cov pytest-xdist
```

### Test Data

The tests use comprehensive fixtures in `fixtures/staking_fixtures.py`:

- **Mock Staking Contract**: Simulates OLAS staking mechanics
- **Compliance Scenarios**: Various compliance test scenarios
- **KPI Test Data**: KPI requirement test data
- **Checkpoint Test Data**: Checkpoint timing test data
- **Vanity Transaction Data**: Vanity transaction test data
- **Test Addresses**: Test addresses for different chains
- **Test Chains**: Optimism and Mode chain configurations

## 📊 Test Results

### Expected Output

```bash
$ pytest tests/integration/protocols/staking/ -v

========================= test session starts =========================
platform linux -- Python 3.10.12, pytest-7.2.1, pluggy-1.0.0
rootdir: /Users/dhairya/Desktop/Work/Valory/Github/optimus
plugins: cov-4.0.0, xdist-3.3.1
collected 50 items

tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_staking_kpi_compliance_workflow PASSED [ 2%]
tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_checkpoint_execution_workflow PASSED [ 4%]
tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_vanity_transaction_workflow PASSED [ 6%]
...
tests/integration/protocols/staking/test_staking_integration.py::TestStakingIntegration::test_complete_end_to_end_workflow PASSED [100%]

========================= 50 passed in 45.67s =========================
```

### Coverage Report

```bash
$ pytest tests/integration/protocols/staking/ --cov=packages.valory.skills.liquidity_trader_abci --cov-report=html

========================= test session starts =========================

## 🐛 Debugging Tests

### Verbose Output

```bash
# Run with verbose output
pytest tests/integration/protocols/staking/ -v -s

# Run with debug logging
pytest tests/integration/protocols/staking/ -v -s --log-cli-level=DEBUG
```

### Specific Test Debugging

```bash
# Run specific test with debug output
pytest tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_staking_kpi_compliance_workflow -v -s --log-cli-level=DEBUG

# Run with pdb for debugging
pytest tests/integration/protocols/staking/test_staking_compliance.py::TestStakingCompliance::test_staking_kpi_compliance_workflow -v -s --pdb
```

### Test Isolation

```bash
# Run tests in isolation
pytest tests/integration/protocols/staking/ --forked

# Run with no warnings
pytest tests/integration/protocols/staking/ --disable-warnings
```

## 🔍 Test Validation

### Verify Test Structure

```bash
# List all test methods
pytest tests/integration/protocols/staking/ --collect-only

# Show test structure
pytest tests/integration/protocols/staking/ --collect-only -q
```

### Verify Fixtures

```bash
# Test fixture loading
pytest tests/integration/protocols/staking/ --fixtures
```

## 📈 Performance Testing

### Parallel Execution

```bash
# Run tests in parallel
pytest tests/integration/protocols/staking/ -n auto

# Run with specific number of workers
pytest tests/integration/protocols/staking/ -n 4
```

### Performance Profiling

```bash
# Run with timing
pytest tests/integration/protocols/staking/ --durations=10

# Run with memory profiling
pytest tests/integration/protocols/staking/ --profile
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Path Issues**: Run from the project root directory
3. **Mock Issues**: Check that mock objects are properly configured
4. **Timeout Issues**: Increase timeout for slow tests

### Debug Commands

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Check imports
python -c "from tests.integration.protocols.staking.fixtures.staking_fixtures import mock_staking_contract; print('Import successful')"

# Check test discovery
pytest tests/integration/protocols/staking/ --collect-only
```

## 📝 Adding New Tests

### Test Template

```python
def test_new_staking_feature(self, mock_staking_contract):
    """Test new staking feature."""
    # Setup
    behaviour = self.create_mock_behaviour(CheckStakingKPIMetBehaviour)
    
    # Mock dependencies
    with patch.object(behaviour, "method") as mock_method:
        mock_method.return_value = "expected_result"
        
        # Execute
        result = behaviour.method()
        
        # Verify
        assert result == "expected_result"
        mock_method.assert_called_once()
```

### Fixture Template

```python
@pytest.fixture
def new_test_data():
    """New test data fixture."""
    return {
        "scenario": "test_scenario",
        "expected_result": "expected_value",
    }
```

## 🎯 Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what they test
2. **Mock External Dependencies**: Use mocks for contract interactions
3. **Test Edge Cases**: Include boundary conditions and error scenarios
4. **Verify Assertions**: Ensure all assertions are meaningful
5. **Clean Up**: Use proper teardown in test methods
6. **Documentation**: Add docstrings to test methods
7. **Isolation**: Tests should be independent and runnable in any order

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Open Autonomy Testing](https://docs.autonolas.network/open-autonomy/develop/test/)
- [AEA Testing Framework](https://open-aea.docs.autonolas.network/develop/test/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
