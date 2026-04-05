# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Optimus is an autonomous agent service built on the **Open Autonomy** framework that manages liquidity provision on the Optimism network. It uses finite state machines (FSMs) to coordinate multi-agent actions across DeFi protocols: Balancer, Uniswap V3, Velodrome, and Merkl.

## Common Commands

### Development Setup
```bash
make poetry-install                    # Install dependencies (includes keyring fix)
autonomy init --reset --author valory --remote --ipfs
autonomy packages sync --update-packages
```

### Testing
```bash
# Run all tests (auto-detects platform)
tox -e py3.10-darwin                   # macOS
tox -e py3.10-linux                    # Linux

# Run a single test directory
pytest -rfE tests/customs/balancer_pools_search
pytest -rfE packages/valory/contracts/uniswap_v3_pool/tests
pytest -rfE packages/valory/skills/liquidity_trader_abci/tests

# Coverage (requires Linux, Python 3.14)
tox -e coverage
```

### Code Quality
```bash
make format                            # Auto-format (black + isort via tomte)
make code-checks                       # Lint + type checks (tomte check-code)
make security                          # bandit + safety + gitleaks
make generators                        # Update ABCI docstrings, copyright headers, package hashes
make common-checks-1                   # copyright, doc links, hash/package checks
make common-checks-2                   # ABCI docstrings, specs, dependencies, handlers
make all-checks                        # Everything above in sequence
```

### FSM Spec Updates
```bash
make fix-abci-app-specs                # Regenerate FSM specs for both skills
autonomy packages lock                 # Rehash packages after changes
```

## Architecture

### FSM Composition (Core Pattern)
The system uses two composed FSMs:

1. **LiquidityTraderAbciApp** (`packages/valory/skills/liquidity_trader_abci/`) — Core trading logic:
   - `FetchStrategiesRound` → `GetPositionsRound` → `EvaluateStrategyRound` → `DecisionMakingRound`
   - Includes staking checkpoint/KPI rounds and withdrawal handling

2. **SuperAgentAbciApp** (`packages/valory/skills/optimus_abci/`) — Wrapper that composes LiquidityTraderAbciApp with transaction settlement, registration, and reset/termination FSMs from the Open Autonomy framework.

### Package Layout under `packages/valory/`
- **skills/** — ABCI skill FSMs (the core logic). Each skill has `rounds.py`, `behaviours.py`, `payloads.py`, `models.py`, and an `fsm_specification.yaml`.
- **contracts/** — Python wrappers around smart contract ABIs for each protocol (balancer_*, uniswap_v3_*, velodrome_*, merkl_*).
- **customs/** — Standalone utility libraries for pool searching (balancer, uniswap, velodrome, merkl), asset lending, and APR selection. Tests for customs live in `tests/customs/` (NOT inside the package directories — customs packages are downloaded from IPFS at runtime by `ComponentPackageLoader`, and including `tests/` directories causes `IsADirectoryError` during IPFS deserialization).
- **connections/** — `mirror_db` connection using Peewee ORM.
- **agents/optimus/** and **services/optimus/** — Agent and service configuration (aea-config.yaml, service.yaml).

### Key Conventions
- Package hashes are tracked in `packages/packages.json` — run `autonomy packages lock` after modifying any package.
- All packages under `packages/valory/` require Valory copyright headers.
- ABCI docstrings and FSM specs are auto-generated — run `make generators` after modifying FSM rounds.
- Tests require `autonomy init` and `autonomy packages sync` before running (handled automatically by tox).
- 100% test coverage is enforced in CI for all code under `packages/valory/`.
- `asyncio_mode=strict` in pytest — use `@pytest.mark.asyncio` explicitly on async tests.

### Code Style
- Black formatting with 88 char line length
- isort with Black-compatible profile
- Darglint docstring linting (Sphinx style, short strictness)
- mypy strict mode targeting Python 3.10
