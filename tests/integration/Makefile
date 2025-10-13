# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

# DeFi Protocol Integration Tests Makefile

.PHONY: help install test test-all test-balancer test-uniswap test-velodrome test-unit test-contract test-e2e test-yield test-transaction test-coverage test-parallel clean

# Default target
help:
	@echo "DeFi Protocol Integration Tests"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install test dependencies"
	@echo "  test             Run all integration tests"
	@echo "  test-all         Run all integration tests (alias for test)"
	@echo "  test-balancer    Run Balancer protocol tests"
	@echo "  test-uniswap     Run Uniswap V3 protocol tests"
	@echo "  test-velodrome   Run Velodrome protocol tests"
	@echo "  test-unit        Run unit integration tests"
	@echo "  test-contract    Run contract integration tests"
	@echo "  test-e2e         Run end-to-end integration tests"
	@echo "  test-yield       Run yield calculation tests"
	@echo "  test-transaction Run transaction generation tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-parallel    Run tests in parallel"
	@echo "  clean            Clean test artifacts"
	@echo ""

# Install test dependencies
install:
	pip install pytest pytest-cov pytest-xdist pytest-html

# Run all integration tests
test test-all:
	python run_integration_tests.py --all

# Run protocol-specific tests
test-balancer:
	python run_integration_tests.py --protocol balancer

test-uniswap:
	python run_integration_tests.py --protocol uniswap_v3

test-velodrome:
	python run_integration_tests.py --protocol velodrome

# Run test type-specific tests
test-unit:
	python run_integration_tests.py --test-type unit

test-contract:
	python run_integration_tests.py --test-type contract

test-e2e:
	python run_integration_tests.py --test-type e2e

test-yield:
	python run_integration_tests.py --test-type yield

test-transaction:
	python run_integration_tests.py --test-type transaction

# Run tests with coverage
test-coverage:
	python run_integration_tests.py --all --coverage

# Run tests in parallel
test-parallel:
	python run_integration_tests.py --all --parallel

# Run tests with verbose output
test-verbose:
	python run_integration_tests.py --all --verbose

# Clean test artifacts
clean:
	rm -rf coverage_html/
	rm -f coverage.xml
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".coverage" -delete

# Quick test runs for development
test-quick:
	python run_integration_tests.py --test-type unit --verbose

test-components:
	python run_integration_tests.py --test-type unit

test-contracts:
	python run_integration_tests.py --test-type contract

test-workflows:
	python run_integration_tests.py --test-type e2e

test-calculations:
	python run_integration_tests.py --test-type yield

test-tx-gen:
	python run_integration_tests.py --test-type transaction

# Development helpers
dev-setup: install
	@echo "Development environment set up for integration tests"

dev-test: test-quick
	@echo "Quick test run completed"

dev-full-test: test-coverage
	@echo "Full test suite with coverage completed"

# CI/CD helpers
ci-test: test-parallel
	@echo "CI test run completed"

ci-coverage: test-coverage
	@echo "CI coverage test run completed"

# Protocol-specific quick tests
quick-balancer:
	python run_integration_tests.py --protocol balancer --test-type unit

quick-uniswap:
	python run_integration_tests.py --protocol uniswap_v3 --test-type unit

quick-velodrome:
	python run_integration_tests.py --protocol velodrome --test-type unit
