## Optimus service

**Supported Chains:**
The Optimus service currently operates on the following chains:
- Optimism
- Base
- Ethereum

**Supported DEXs:**
The Optimus service trades on the following decentralized exchanges (DEXs):
- Balancer
- Uniswap

**Operational Process:**
Within a typical epoch of 24 hours, the Optimus service performs the following tasks:

1. **Opportunity Identification:** It identifies trading opportunities through campaigns advertised on the Merkl platform across the supported DEXs.
  
2. **Liquidity Pool Investment:** When a suitable liquidity pool is found, the service takes the following actions:
   - **First Opportunity:** If this is the first opportunity encountered, the service will add liquidity to the pool if its Annual Percentage Rate (APR) is higher than 5%.
   - **Subsequent Opportunities:** If the service has already invested in a liquidity pool, it will consider the next opportunity only if its APR exceeds that of the previously invested pool.

3. **Transaction Tracking:** The service tracks the number of transactions performed on the Optimism chain, which serves as key performance indicators (KPIs) for Olas Staking Rewards.

The Optimus service is an [agent service](https://docs.autonolas.network/open-autonomy/get_started/what_is_an_agent_service/) (or autonomous service) based on the [Open Autonomy framework](https://docs.autonolas.network/open-autonomy/). Below we show you how to prepare your environment, how to prepare the agent keys, and how to configure and run the service.

## Prepare the environment

- System requirements:

  - Python `== 3.10`
  - [Poetry](https://python-poetry.org/docs/) `>=1.4.0`
  - [Docker Engine](https://docs.docker.com/engine/install/)
  - [Docker Compose](https://docs.docker.com/compose/install/)

- Clone this repository:

      git clone https://github.com/valory-xyz/optimus.git

- Create a development environment:

      poetry install && poetry shell

- Configure the Open Autonomy framework:

      autonomy init --reset --author valory --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"

- Pull packages required to run the service:

      autonomy packages sync --update-packages

## Prepare the keys and the Safe

1. You need a **Gnosis keypair** to run the service.

First, prepare the `keys.json` file with the Gnosis keypair of your agent. (Replace the uppercase placeholders below):

    cat > keys.json << EOF
    [
    {
        "address": "YOUR_AGENT_ADDRESS",
        "private_key": "YOUR_AGENT_PRIVATE_KEY"
    }
    ]
    EOF

2. You need to deploy 4 **[Safes](https://safe.global/) on the following networks - Ethereum-Mainnet, Optimism, Base, Mode**

3. You need to provide some funds ETH and USDC both to your Safe address on Ethereum-Mainnet, and some ETH to your agent across all the chains (Ethereum-Mainnet, Optimism, Base, Mode) to cover for gas fees.

4. You will need your Tenderly Access Key, Tenderly account Slug, and Tenderly Project Slug. Get one at https://dashboard.tenderly.co/ under settings.

5. You will need also need Coingecko API Key. Get one at https://www.coingecko.com/ under My Account -> Developer's Dashboard.

## Configure the service

Set up the following environment variables, to run the service. **Please read their description below**.

```bash
export ETHEREUM_LEDGER_RPC=INSERT_YOUR_ETHEREUM_RPC
export OPTIMISM_LEDGER_RPC=INSERT_YOUR_OPTIMISM_RPC
export BASE_LEDGER_RPC=INSERT_YOUR_BASE_RPC
export MODE_LEDGER_RPC=INSERT_YOUR_MODE_RPC

export ALL_PARTICIPANTS='["YOUR_AGENT_ADDRESS"]'
export SAFE_CONTRACT_ADDRESSES='{ethereum:"YOUR_SAFE_ADDRESS_ON_ETHEREUM","optimism":"YOUR_SAFE_ADDRESS_ON_OPTIMISM","base":"YOUR_SAFE_ADDRESS_ON_BASE","mode":"YOUR_SAFE_ADDRESS_ON_MODE"}'

export SLIPPAGE_FOR_SWAP=0.09
export TENDERLY_ACCESS_KEY=YOUR_TENDERLY_ACCESS_KEY
export TENDERLY_ACCOUNT_SLUG=YOUR_TENDERLY_ACCOUNT_SLUG
export TENDERLY_PROJECT_SLUG=YOUR_TENDERLY_PROJECT_SLUG
export COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY

Note - We provide default value for SLIPPAGE_FOR_SWAP, but feel free to experiment with different values. It indicates the allowed slippage when bridging/swapping assets using LiFi.

## Run the service
Once you have configured (exported) the environment variables, you are in position to run the service.

- Fetch the service:

    ```bash
    autonomy fetch --local --service valory/optimus && cd optimus
    ```

- Build the Docker image:

    ```bash
    autonomy build-image
    ```

- Copy your `keys.json` file prepared [in the previous section](#prepare-the-keys-and-the-safe) in the same directory:

    ```bash
    cp path/to/keys.json .
    ```

- Build the deployment with a single agent and run:

    ```bash
    autonomy deploy build --n 1 -ltm
    autonomy deploy run --build-dir abci_build/
    ```
