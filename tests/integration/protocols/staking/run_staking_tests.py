#!/usr/bin/env python3
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

"""Staking compliance test runner script."""

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


def run_staking_tests(
    test_file: Optional[str] = None,
    test_method: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False,
    debug: bool = False
) -> int:
    """Run staking compliance tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory or specific file
    test_dir = Path(__file__).parent
    if test_file:
        cmd.append(str(test_dir / test_file))
    else:
        cmd.append(str(test_dir))
    
    # Add specific test method
    if test_method:
        cmd.extend(["-k", test_method])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add debug flag
    if debug:
        cmd.extend(["-s", "--log-cli-level=DEBUG"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=packages.valory.skills.liquidity_trader_abci",
            "--cov-report=html:coverage_html",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term"
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
    parser = argparse.ArgumentParser(description="Run staking compliance tests")
    
    parser.add_argument(
        "--test-file",
        choices=[
            "test_staking_compliance.py",
            "test_checkpoint_mechanisms.py", 
            "test_vanity_transactions.py",
            "test_kpi_compliance.py",
            "test_staking_integration.py"
        ],
        help="Run tests for specific file only"
    )
    
    parser.add_argument(
        "--test-method",
        help="Run specific test method (use partial name matching)"
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
        "--debug",
        action="store_true",
        help="Debug mode with detailed logging"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all staking tests"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (exclude slow integration tests)"
    )
    
    args = parser.parse_args()
    
    # If no specific options, run all tests
    if not any([args.test_file, args.test_method, args.all, args.quick]):
        print("No specific test selection. Use --all to run all tests or specify --test-file/--test-method")
        return 1
    
    # Run tests
    exit_code = run_staking_tests(
        test_file=args.test_file,
        test_method=args.test_method,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        debug=args.debug
    )
    
    if exit_code == 0:
        print("\n✅ All staking compliance tests passed!")
    else:
        print("\n❌ Some staking compliance tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
