# Balancer Pool Analysis Tools

## Overview
This repository contains three scripts designed to fetch and analyze liquidity metrics from Balancer pools using the Balancer V3 GraphQL API. These tools are essential for understanding the liquidity dynamics, volume, and potential risks associated with trading in specific pools on Balancer.

## Scripts

### 1. Fetch Liquidity Metrics
**Purpose**: This script fetches liquidity metrics such as total liquidity and 24-hour volume for a specified pool over the past 90 days.

**Output**: The output includes timestamped data on total liquidity and 24-hour volume.

**Requirements**:
- Python 3.x
- `gql` library for GraphQL queries

**Usage**:
- Set up the required GraphQL client and define the pool ID.
- Execute the script to print liquidity metrics in the console.

### 2. Fetch Pool Snapshots
**Purpose**: Similar to the first script, but with additional details, this script fetches a range of data snapshots for a specified pool, including its ID, total liquidity, and total swap volume.

**Output**: The output is printed directly in the console and includes comprehensive pool snapshot data.

**Requirements**:
- Python 3.x
- `gql` library

**Usage**:
- Configure the GraphQL client with the correct API endpoint.
- Execute the script to retrieve and print data snapshots.

### 3. Balancer Liquidity Analyzer
**Purpose**: This comprehensive script not only fetches data but also performs detailed analysis on liquidity metrics, providing insights into depth score, liquidity risk, and maximum position size for a given Balancer pool.

**Output**: Provides a detailed report on average total value locked (TVL), daily volume, liquidity depth score, and liquidity risk assessment.

**Requirements**:
- Python 3.x
- `gql` library
- `statistics` library for data analysis

**Usage**:
- Instantiate the `BalancerLiquidityAnalyzer` with the necessary API URL.
- Call the `analyze_pool_data` function with the pool ID to generate a detailed liquidity analysis report.

## Installation
To run these scripts, you will need to install the required Python libraries. You can install these dependencies via pip:

```bash
pip install gql statistics
```

## Configuration
Before running the scripts, ensure the API URL (`https://api-v3.balancer.fi`) is correctly configured in the scripts to match the endpoint provided by Balancer.

## Running the Scripts
To execute any of the scripts, navigate to the directory containing the script and run:

```bash
python <script_name>.py
```

Replace `<script_name>` with the name of the script you wish to run.

