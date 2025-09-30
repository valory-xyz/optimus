.PHONY: clean
clean: clean-test clean-build clean-pyc clean-docs

.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr pip-wheel-metadata
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -type d -name __pycache__ -exec rm -rv {} +
	rm -fr Pipfile.lock
	rm -rf plugins/*/build
	rm -rf plugins/*/dist

.PHONY: clean-docs
clean-docs:
	rm -fr site/

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr .tox/
	rm -f .coverage
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .hypothesis
	rm -fr .pytest_cache
	rm -fr .mypy_cache/
	rm -fr .hypothesis/
	find . -name 'log.txt' -exec rm -fr {} +
	find . -name 'log.*.txt' -exec rm -fr {} +

# isort: fix import orders
# black: format files according to the pep standards
.PHONY: formatters
formatters:
	tomte format-code

# black-check: check code style
# isort-check: check for import order
# flake8: wrapper around various code checks, https://flake8.pycqa.org/en/latest/user/error-codes.html
# mypy: static type checker
# pylint: code analysis for code smells and refactoring suggestions
# darglint: docstring linter
.PHONY: code-checks
code-checks:
	tomte check-code

# safety: checks dependencies for known security vulnerabilities
# bandit: security linter
.PHONY: security
security:
	tomte check-security
	gitleaks detect --report-format json --report-path leak_report

# generate latest hashes for updated packages
# generate docs for updated packages
# update copyright headers
.PHONY: generators
generators:
	tox -e abci-docstrings
	tomte format-copyright --author author_name
	autonomy packages lock

.PHONY: common-checks-1
common-checks-1:
	tomte check-copyright --author author_name
	tomte check-doc-links
	tox -p -e check-hash -e check-packages -e check-doc-hashes

.PHONY: all-linters
all-linters:
	gitleaks detect --report-format json --report-path leak_report
	tox -e spell-check
	tox -e check-doc-hashes
	tox -e bandit
	tox -e safety
	tox -e liccheck
	tox -e check-packages
	tox -e check-abciapp-specs
	tox -e check-hash
	tox -e black-check
	tox -e isort-check
	tox -e flake8
	tox -e darglint
	tox -e pylint
	tox -e mypy

.PHONY: fix-abci-app-specs
fix-abci-app-specs:
	export PYTHONPATH=${PYTHONPATH}:${PWD}
	autonomy analyse fsm-specs --update --app-class LiquidityTraderAbciApp --package packages/valory/skills/liquidity_trader_abci/ || (echo "Failed to check liquidity_trader_abci abci consistency" && exit 1)
	autonomy analyse fsm-specs --update --app-class SuperAgentAbciApp --package packages/valory/skills/optimus_abci/ || (echo "Failed to check optimus_abci abci consistency" && exit 1)


.PHONY: tm
tm:
	rm -r ~/.tendermint
	tendermint init
	tendermint node --proxy_app=tcp://127.0.0.1:26658 --rpc.laddr=tcp://127.0.0.1:26657 --p2p.laddr=tcp://0.0.0.0:26656 --p2p.seeds= --consensus.create_empty_blocks=true

v := $(shell pip -V | grep virtualenvs)

.PHONY: poetry-install
poetry-install: 

	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry run pip install --upgrade --force-reinstall setuptools==59.5.0  # fix for KeyError: 'setuptools._distutils.compilers'



./agent:  poetry-install ./hash_id
	@if [ ! -d "agent" ]; then \
		poetry run autonomy -s fetch --remote `cat ./hash_id` --alias agent; \
	fi \


.PHONY: build-agent-runner
build-agent-runner: poetry-install agent
	poetry run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--collect-all aea_ledger_ethereum_flashbots \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum_flashbots \
	$(shell poetry run python get_pyinstaller_dependencies.py) \
	--onefile pyinstaller/optimus_bin.py \
	--name agent_runner_bin
	./dist/agent_runner_bin --version
	

.PHONY: build-agent-runner-mac
build-agent-runner-mac: poetry-install  agent
	poetry run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--collect-all aea_ledger_ethereum_flashbots \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum_flashbots \
	$(shell poetry run python get_pyinstaller_dependencies.py) \
	--onefile pyinstaller/optimus_bin.py \
	--codesign-identity "${SIGN_ID}" \
	--name agent_runner_bin
	./dist/agent_runner_bin --version


./hash_id: ./packages/packages.json
	cat ./packages/packages.json | jq -r '.dev | to_entries[] | select(.key | startswith("agent/")) | .value' > ./hash_id

./agent_id: ./packages/packages.json
	cat ./packages/packages.json | jq -r '.dev | to_entries[] | select(.key | startswith("agent/")) | .key | sub("^agent/"; "")' > ./agent_id

./agent.zip: ./agent
	zip -r ./agent.zip ./agent

./agent.tar.gz: ./agent
	tar czf ./agent.tar.gz ./agent

./agent/ethereum_private_key.txt: ./agent
	poetry run bash -c "cd ./agent; autonomy  -s generate-key ethereum; autonomy  -s add-key ethereum ethereum_private_key.txt; autonomy -s issue-certificates;"


# Configuration
TIMEOUT := 20
COMMAND := cd ./agent && SKILL_TRADER_ABCI_MODELS_PARAMS_ARGS_STORE_PATH=/tmp ../dist/agent_runner_bin -s run
SEARCH_STRING := Starting AEA


# Determine OS and set appropriate options
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS specific settings
    MKTEMP = mktemp -t tmp
else ifeq ($(OS),Windows_NT)
    # Windows specific settings
    MKTEMP = echo $$(cygpath -m "$$(mktemp -t tmp.XXXXXX)")
else
    # Linux and other Unix-like systems
    MKTEMP = mktemp
endif

.PHONY: check-agent-runner
check-agent-runner:
	python check_agent_runner.py