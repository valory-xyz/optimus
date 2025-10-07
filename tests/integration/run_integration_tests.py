#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
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

"""Integration test runner script."""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return the exit code."""
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return 1


def run_integration_tests(
    protocol: Optional[str] = None,
    test_type: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False
) -> int:
    """Run integration tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add protocol filter
    if protocol:
        cmd.extend(["-m", protocol])
    
    # Add test type filter
    if test_type:
        cmd.extend(["-m", test_type])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=packages.valory.skills.liquidity_trader_abci",
            "--cov-report=html:coverage_html",
            "--cov-report=xml:coverage.xml"
        ])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    return run_command(cmd)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DeFi protocol integration tests")
    
    parser.add_argument(
        "--protocol",
        choices=["balancer", "uniswap_v3", "velodrome"],
        help="Run tests for specific protocol only"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["unit", "contract", "e2e", "yield", "transaction"],
        help="Run specific type of tests only"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all integration tests"
    )
    
    args = parser.parse_args()
    
    # If no specific options, run all tests
    if not any([args.protocol, args.test_type, args.all]):
        print("No specific test selection. Use --all to run all tests or specify --protocol/--test-type")
        return 1
    
    # Run tests
    exit_code = run_integration_tests(
        protocol=args.protocol,
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
