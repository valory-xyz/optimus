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

import json
import os


ROUNDS_INFO = {
    "APRPopulationRound": {
        "name": "Calculating APR",
        "description": "Calculates APR and other relevant trading performance of the agent and writes it to database",
        "transitions": {},
    },
    "CallCheckpointRound": {
        "name": "Checking checkpoint",
        "description": "Decides if the staking contract should be signaled for rewards.",
        "transitions": {},
    },
    "CheckLateTxHashesRound": {
        "name": "Checking late transaction hashes",
        "description": "Determining if ok to proceed.",
        "transitions": {},
    },
    "CheckStakingKPIMetRound": {
        "name": "Checking staking requirements",
        "description": "Checks if the agent is on track to earn staking rewards.",
        "transitions": {},
    },
    "CheckTransactionHistoryRound": {
        "name": "Checking the transaction history",
        "description": "Checks the transaction history to see if any previous transaction has been validated",
        "transitions": {},
    },
    "CollectSignatureRound": {
        "name": "Collecting agent signatures",
        "description": "Agent signs the transaction.",
        "transitions": {},
    },
    "DecisionMakingRound": {
        "name": "Executing the actions",
        "description": "Executes trades on-chain.",
        "transitions": {},
    },
    "EvaluateStrategyRound": {
        "name": "Evaluating the strategies",
        "description": "Evaluates strategies and decides the best course of action for the agent.",
        "transitions": {},
    },
    "FetchStrategiesRound": {
        "name": "Loading trading strategies",
        "description": "Curates the agent portfolio.",
        "transitions": {},
    },
    "FinalizationRound": {
        "name": "Submitting transaction",
        "description": "Submits a transaction to the blockchain.",
        "transitions": {},
    },
    "GetPositionsRound": {
        "name": "Checking balances",
        "description": "Identifies the agent's positions and checks the balance of its assets.",
        "transitions": {},
    },
    "PostTxSettlementRound": {
        "name": "Deciding the next round",
        "description": "Reviews activity and prepares hand-off for next round.",
        "transitions": {},
    },
    "RandomnessTransactionSubmissionRound": {
        "name": "Generating randomness",
        "description": "Prepares the agent to sign the transaction.",
        "transitions": {},
    },
    "RegistrationRound": {
        "name": "Registering agents",
        "description": "Organises agent components.",
        "transitions": {},
    },
    "RegistrationStartupRound": {
        "name": "Registering agents at startup",
        "description": "Prepares for startup.",
        "transitions": {},
    },
    "ResetAndPauseRound": {
        "name": "Preparing for next cycle",
        "description": "Cleans up and takes a break before starting the next trading cycle",
        "transitions": {},
    },
    "ResetRound": {
        "name": "Cleaning up and resetting",
        "description": "Clean ups and prepares the next round.",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionARound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Agent is selected for transaction submission",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBAfterTimeoutRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Agent is selected for transaction submission",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Agent is selected for transaction submission",
        "transitions": {},
    },
    "SynchronizeLateMessagesRound": {
        "name": "Synchronizing late messages",
        "description": "Checks that the submitted transaction was successful.",
        "transitions": {},
    },
    "ValidateTransactionRound": {
        "name": "Validating the transaction",
        "description": "Checks that the submitted transaction was successful.",
        "transitions": {},
    },
    "WithdrawFundsRound": {
        "name": "Withdrawing funds",
        "description": "Withdraws funds from the agent",
        "transitions": {},
    },
}
