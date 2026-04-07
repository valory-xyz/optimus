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
clean-test: clean-cache
	rm -fr .tox/
	rm -f .coverage
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;
	rm -fr coverage.xml
	rm -fr htmlcov/
	find . -name 'log.txt' -exec rm -fr {} +
	find . -name 'log.*.txt' -exec rm -fr {} +
	rm -rf leak_report

# removes various cache files
.PHONY: clean-cache
clean-cache:
	find . -type d -name .hypothesis -prune -exec rm -rf {} \;
	rm -fr .pytest_cache
	rm -fr .mypy_cache/

# isort: fix import orders
# black: format files according to the pep standards
.PHONY: formatters
formatters:
	tomte format-code

.PHONY: format
format:
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
# gitleaks: checks for sensitive information
.PHONY: security
security:
	tomte check-security
	gitleaks detect --report-format json --report-path leak_report --log-opts="HEAD"

# generate abci docstrings
# update copyright headers
# generate latest hashes for updated packages
.PHONY: generators
generators: clean-cache fix-abci-app-specs
	tox -qq -e abci-docstrings
	tomte format-copyright --author valory --exclude-part signing --exclude-part acn --exclude-part http --exclude-part ledger_api --exclude-part contract_api --exclude-part abci --exclude-part tendermint --exclude-part ipfs --exclude-part srr --exclude-part kv_store --exclude-part gnosis_safe_proxy_factory --exclude-part service_registry --exclude-part gnosis_safe --exclude-part multisend --exclude-part erc20 --exclude-part staking_activity_checker --exclude-part staking_token --exclude-part http_client --exclude-part ledger --exclude-part p2p_libp2p_client --exclude-part http_server --exclude-part x402 --exclude-part genai --exclude-part abstract_abci --exclude-part abstract_round_abci --exclude-part transaction_settlement_abci --exclude-part registration_abci --exclude-part reset_pause_abci --exclude-part termination_abci --exclude-part funds_manager
	autonomy packages lock

.PHONY: common-checks-1
common-checks-1:
	tox -qq -e copyright-check
	tomte check-doc-links
	tox -qq -p -e check-hash -e check-packages -e check-doc-hashes

.PHONY: common-checks-2
common-checks-2:
	tox -qq -e check-abci-docstrings
	tox -qq -e check-abciapp-specs
	tox -qq -e check-dependencies
	tox -qq -e check-handlers

.PHONY: all-checks
all-checks: format code-checks security generators common-checks-1 common-checks-2

.PHONY: ci-linter-checks
ci-linter-checks:
	gitleaks detect --report-format json --report-path leak_report --log-opts="HEAD"
	tox -qq -e copyright-check
	tox -qq -e liccheck
	tox -qq -e check-dependencies
	tomte check-doc-links
	tox -qq -e check-doc-hashes
	tomte check-security
	tox -qq -e check-packages
	tox -qq -e check-hash
	tomte check-code
	tomte check-spelling
	tox -qq -e check-abci-docstrings
	tox -qq -e check-abciapp-specs
	tox -qq -e check-handlers

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

.PHONY: uv-install
uv-install:
	uv sync --all-groups

./agent:  uv-install ./hash_id
	@if [ ! -d "agent" ]; then \
		uv run autonomy -s fetch --remote `cat ./hash_id` --alias agent; \
	fi \


.PHONY: build-agent-runner
build-agent-runner: uv-install agent
	uv run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--collect-all aea_ledger_ethereum_flashbots \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum_flashbots \
	$(shell uv run aea-helpers build-binary-deps ./agent) \
	--onefile $(shell uv run aea-helpers bin-template-path) \
	--name agent_runner_bin
	./dist/agent_runner_bin --version


.PHONY: build-agent-runner-mac
build-agent-runner-mac: uv-install  agent
	uv run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--collect-all aea_ledger_ethereum_flashbots \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum_flashbots \
	$(shell uv run aea-helpers build-binary-deps ./agent) \
	--onefile $(shell uv run aea-helpers bin-template-path) \
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
	uv run bash -c "cd ./agent; autonomy  -s generate-key ethereum; autonomy  -s add-key ethereum ethereum_private_key.txt; autonomy add-key ethereum ethereum_private_key.txt --connection; autonomy -s issue-certificates;"


.PHONY: check-agent-runner
check-agent-runner:
	uv run aea-helpers check-binary ./dist/agent_runner_bin ./agent \
	--env-var SKILL_OPTIMUS_ABCI_MODELS_PARAMS_ARGS_STORE_PATH=/tmp \
	--env-var CONNECTION_KV_STORE_CONFIG_STORE_PATH=/tmp