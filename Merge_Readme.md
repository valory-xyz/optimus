# Run Your Own Agent

This guide will help you set up and run your own agent, either **Optimus** or **BabyDegen**. Follow the steps below to get started.

---

## Table of Contents

1. [Get the Code](#1-get-the-code)
2. [Set Up the Virtual Environment](#2-set-up-the-virtual-environment)
3. [Synchronize Packages](#3-synchronize-packages)
4. [Prepare the Data](#4-prepare-the-data)
5. [Configure for Optimus](#5-configure-for-optimus)
6. [Configure for BabyDegen](#6-configure-for-babydegen)
7. [Run the Agent](#7-run-the-agent)

---

## 1. Get the Code

Clone the repository from GitHub:

```bash
git clone https://github.com/valory-xyz/optimus.git
```

---

## 2. Set Up the Virtual Environment

Navigate to the project directory and install the required dependencies using `poetry`:

```bash
cd optimus
poetry install
poetry shell
```

---

## 3. Synchronize Packages

Synchronize the necessary packages:

```bash
autonomy packages sync --update-packages
```

---

## 4. Prepare the Data

### Generate Wallet Keys

Create a `keys.json` file containing wallet addresses and private keys for four agents:

```bash
autonomy generate-key ethereum -n 4
```

### Create Ethereum Private Key File

Extract one of the private keys from `keys.json` and save it in a file named `ethereum_private_key.txt`. Ensure there's **no newline at the end of the file**.

---

## 5. Configure for Optimus

If you want to run the **Optimus** agent, follow these steps:

### a. Deploy Safe Contracts

Deploy [Safe](https://safe.global/) contracts on the following networks:

- Ethereum Mainnet
- Optimism
- Base
- Mode

### b. Fund Your Safe and Agent Addresses

- **Safe Addresses**:
  - Deposit **ETH** and **USDC** into your Safe address on **Ethereum Mainnet**.
- **Agent Addresses**:
  - Deposit **ETH** into your agent addresses on all networks (Ethereum Mainnet, Optimism, Base, Mode) to cover gas fees.

### c. Obtain API Keys

- **Tenderly**:
  - Access Key
  - Account Slug
  - Project Slug
  - Get them from your [Tenderly Dashboard](https://dashboard.tenderly.co/) under **Settings**.
- **CoinGecko**:
  - API Key
  - Obtain it from your account's [Developer Dashboard](https://www.coingecko.com/account/dashboard).

### d. Set Environment Variables

Replace placeholder values with your actual data:

```bash
export ETHEREUM_LEDGER_RPC=YOUR_ETHEREUM_RPC_URL
export OPTIMISM_LEDGER_RPC=YOUR_OPTIMISM_RPC_URL
export BASE_LEDGER_RPC=YOUR_BASE_RPC_URL

export ALL_PARTICIPANTS='["YOUR_AGENT_ADDRESS"]'
export SAFE_CONTRACT_ADDRESSES='{
  "ethereum": "YOUR_SAFE_ADDRESS_ON_ETHEREUM",
  "optimism": "YOUR_SAFE_ADDRESS_ON_OPTIMISM",
  "base": "YOUR_SAFE_ADDRESS_ON_BASE",
  "mode": "YOUR_SAFE_ADDRESS_ON_MODE"
}'

export SLIPPAGE_FOR_SWAP=0.09
export TENDERLY_ACCESS_KEY=YOUR_TENDERLY_ACCESS_KEY
export TENDERLY_ACCOUNT_SLUG=YOUR_TENDERLY_ACCOUNT_SLUG
export TENDERLY_PROJECT_SLUG=YOUR_TENDERLY_PROJECT_SLUG
export COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY
```

---

## 6. Configure for BabyDegen

If you prefer to run the **BabyDegen** agent, follow these additional steps:

### a. Set the AGENT_TRANSITION Flag

Set the `AGENT_TRANSITION` flag to `true` in your environment or configuration files.

### b. Update Safe Address and Participants

Replace placeholders with your actual Safe address and agent address:

```bash
export SAFE_ADDRESS="YOUR_SAFE_ADDRESS"
export ALL_PARTICIPANTS='["YOUR_AGENT_ADDRESS"]'
```

### c. Fund the Safe Account

Deposit the following tokens into your Safe account, depending on the network:

```python
LEDGER_TO_TOKEN_LIST = {
    SupportedLedgers.ETHEREUM: [
        "0x0001a500a6b18995b03f44bb040a5ffc28e45cb0",  # Token A on Ethereum
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC on Ethereum
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH on Ethereum
    ],
    SupportedLedgers.OPTIMISM: [
        "0x4200000000000000000000000000000000000006",  # WETH on Optimism
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85",  # Token B on Optimism
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI on Optimism
    ],
    SupportedLedgers.BASE: [
        "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca",  # Token C on Base
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",  # Token D on Base
    ],
}
```

### d. Fund the Agent Address

Ensure your agent address has enough native tokens to cover gas fees on your chosen network.

---

## 7. Run the Agent

After completing the setup for either **Optimus** or **BabyDegen**, run the agent using the provided script:

```bash
bash run_agent.sh
```
