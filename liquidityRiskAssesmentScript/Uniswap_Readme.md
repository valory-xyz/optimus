# Liquidity Analysis Tools for Uniswap Ethereum DeFi Pools

This repository contains Python scripts that leverage The Graph API to fetch and analyze data from Uniswap decentralized finance (DeFi) protocols on Ethereum. These tools are critical for evaluating pool dynamics, volume, and associated risks.

## Scripts

### 1. `fetch_pool_data.py`
**Purpose**: This script retrieves basic liquidity and token information for the top 1000 pools by liquidity.

**Output**:
- Displays pool IDs, liquidity, and token details for the top pools.

**Requirements**:
- Python 3.x
- `requests` library

**Usage**:
- Execute the script to print detailed data about the top liquidity pools directly to the console.

### 2. `fetch_24_hour_volume.py`
**Purpose**: Fetches the 24-hour trading volume data for pools, providing insights into recent trading activity.

**Output**:
- Outputs the 24-hour volume for pools, along with liquidity and activity ratios.

**Requirements**:
- Python 3.x
- `requests` library
- `time` library for timestamp calculations

**Usage**:
- Run the script to obtain and display the latest 24-hour trading volume data for the top pools.

### 3. `assess_liquidity_risk.py`
**Purpose**: Combines data fetched by the previous scripts to assess liquidity risk, considering factors like total value locked, volume, and transaction count.

**Output**:
- Provides a detailed report on average total value locked (TVL), daily volume, liquidity depth score, and liquidity risk assessment.

**Requirements**:
- Python 3.x
- `requests` library

**Usage**:
- Execute to perform a comprehensive liquidity risk analysis on major DeFi pools and print the results.

## Installation
Install the necessary Python libraries to run the scripts. You can do this using pip:

```bash
pip install requests
```

## Configuration
Before running the scripts, ensure you have a valid API key for The Graph API and that it is correctly set up in the scripts. Replace `API_KEY` with your actual key in the scripts.

## Running the Scripts
Navigate to the directory containing the script and run:

```bash
python <script_name>.py
```

Replace `<script_name>` with the name of the script you want to execute, such as `fetch_pool_data.py`.

## Notes
- Ensure you are connected to the internet as these scripts require real-time data fetching from The Graph API.
- Handle private keys and API keys with care, ensuring they are not exposed in public repositories or shared environments.

