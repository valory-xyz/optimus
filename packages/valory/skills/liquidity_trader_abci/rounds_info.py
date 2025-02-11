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

"""This module contains the information about the rounds that is used by the Http handler."""


ROUNDS_INFO = {
    "CallCheckpointRound": {
        "name": "Checking checkpoint",
        "description": "Decides if a checkpoint transaction should be made in the staking contract.",
        "transitions": {},
    },
    "CheckLateTxHashesRound": {
        "name": "Checking late transaction hashes",
        "description": "Checks late transaction hashes",
        "transitions": {},
    },
    "CheckStakingKPIMetRound": {
        "name": "Checking staking kpi",
        "description": "Checks if the KPI for earning staking rewards is met; if not, ensures compliance by performing vanity transactions.",
        "transitions": {},
    },
    "CheckTransactionHistoryRound": {
        "name": "Checking the transaction history",
        "description": "Checks the transaction history",
        "transitions": {},
    },
    "CollectSignatureRound": {
        "name": "Collecting agent signatures",
        "description": "Collects agent signatures for a transaction",
        "transitions": {},
    },
    "DecisionMakingRound": {
        "name": "Executing the actions",
        "description": "Executes all actions required by the agent.",
        "transitions": {},
    },
    "EvaluateStrategyRound": {
        "name": "Evaluating the strategies",
        "description": "Evaluates all strategies and decides the best course of action for the agent.",
        "transitions": {},
    },
    "FinalizationRound": {
        "name": "Sending a transaction",
        "description": "Sends a transaction for mining",
        "transitions": {},
    },
    "GetPositionsRound": {
        "name": "Checking the balances",
        "description": "Identifies the service's positions and checks the balance of its assets.",
        "transitions": {},
    },
    "PostTxSettlementRound": {
        "name": "Deciding the next round",
        "description": "Transitions to the correct round after a transaction is settled via the transaction_settlement_abci, based on the previous round and event.",
        "transitions": {},
    },
    "RandomnessTransactionSubmissionRound": {
        "name": "Getting some randomness",
        "description": "Gets randomness from a decentralized randomness source",
        "transitions": {},
    },
    "RegistrationRound": {
        "name": "Registering agents ",
        "description": "Initializes the agent registration process",
        "transitions": {},
    },
    "RegistrationStartupRound": {
        "name": "Registering agents at startup",
        "description": "Initializes the agent registration process",
        "transitions": {},
    },
    "ResetAndPauseRound": {
        "name": "Cleaning up and sleeping for some time",
        "description": "Cleans up and sleeps for some time before running again",
        "transitions": {},
    },
    "ResetRound": {
        "name": "Cleaning up and resetting",
        "description": "Cleans up and resets the agent",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionARound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBAfterTimeoutRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SynchronizeLateMessagesRound": {
        "name": "Synchronizing late messages",
        "description": "Synchronizes late messages",
        "transitions": {},
    },
    "TransactionMultiplexerRound": {
        "name": "Selecting next round",
        "description": "Decides where to transition next based on the state previous to the transaction",
        "transitions": {},
    },
    "ValidateTransactionRound": {
        "name": "Validating the transaction",
        "description": "Checks that the transaction was succesful",
        "transitions": {},
    },
    "BacktestRound": {
        "name": "Backtest",
        "description": "Runs a backtest of the trading strategy",
        "transitions": {},
    },
    "DecideAgentEndingRound": {
        "name": "Deciding agent ending",
        "description": "Decides when the agent should end its execution",
        "transitions": {},
    },
    "DecideAgentStartingRound": {
        "name": "Deciding agent starting",
        "description": "Decides when the agent should start its execution",
        "transitions": {},
    },
    "FetchMarketDataRound": {
        "name": "Fetching market data",
        "description": "Fetches market data from external sources",
        "transitions": {},
    },
    "PortfolioTrackerRound": {
        "name": "Tracking the portfolio",
        "description": "Tracks the agent's portfolio and positions",
        "transitions": {},
    },
    "PrepareEvmSwapRound": {
        "name": "Preparing EVM swap",
        "description": "Prepares the necessary data for an EVM swap transaction",
        "transitions": {},
    },
    "PrepareSwapRound": {
        "name": "Preparing swap",
        "description": "Prepares the necessary data for a swap transaction",
        "transitions": {},
    },
    "ProxySwapQueueRound": {
        "name": "Proxy swap queue",
        "description": "Manages the queue of swap transactions",
        "transitions": {},
    },
    "RandomnessRound": {
        "name": "Getting randomness",
        "description": "Obtains randomness from a decentralized source",
        "transitions": {},
    },
    "StrategyExecRound": {
        "name": "Executing the strategy",
        "description": "Executes the trading strategy",
        "transitions": {},
    },
    "SwapQueueRound": {
        "name": "Swap queue",
        "description": "Manages the queue of swap transactions",
        "transitions": {},
    },
    "TraderDecisionMakerRound": {
        "name": "Trader decision maker",
        "description": "Makes trading decisions based on the evaluated strategies",
        "transitions": {},
    },
    "TransformMarketDataRound": {
        "name": "Transforming market data",
        "description": "Transforms the fetched market data into a format suitable for the trading strategy",
        "transitions": {},
    },
}
