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
    "APRPopulationRound": {
        "name": "Updating performance",
        "description": "Calculates the agent’s APR and overall trading performance, then saves it to your dashboard.",
        "transitions": {},
    },
    "CallCheckpointRound": {
        "name": "Checking staking reward status",
        "description": "Determines whether the staking contract needs to update the agent’s reward status.",
        "transitions": {},
    },
    "CheckLateTxHashesRound": {
        "name": "Reviewing pending transactions",
        "description": "Checks if any earlier transactions are still pending before moving forward.",
        "transitions": {},
    },
    "CheckStakingKPIMetRound": {
        "name": "Checking staking requirements",
        "description": "Ensures the agent is on track to meet the conditions needed to earn staking rewards.",
        "transitions": {},
    },
    "CheckTransactionHistoryRound": {
        "name": "Reviewing transaction history",
        "description": "Looks at previous transactions to confirm they were processed correctly.",
        "transitions": {},
    },
    "CollectSignatureRound": {
        "name": "Signing the transaction",
        "description": "The agent signs the transaction so it’s ready to be submitted.",
        "transitions": {},
    },
    "DecisionMakingRound": {
        "name": "Executing trades",
        "description": "The agent performs on-chain trades according to its strategy.",
        "transitions": {},
    },
    "EvaluateStrategyRound": {
        "name": "Evaluating strategies",
        "description": "The agent reviews its available strategies and chooses the best next move.",
        "transitions": {},
    },
    "FetchStrategiesRound": {
        "name": "Loading trading strategies",
        "description": "The agent updates and organizes the strategies that guide its portfolio decisions.",
        "transitions": {},
    },
    "FinalizationRound": {
        "name": "Submitting the transaction",
        "description": "The agent sends the prepared transaction to the blockchain.",
        "transitions": {},
    },
    "GetPositionsRound": {
        "name": "Checking portfolio",
        "description": "The agent reviews its positions and checks the balances of all assets it manages.",
        "transitions": {},
    },
    "PostTxSettlementRound": {
        "name": "Reviewing results",
        "description": "The agent reviews its activity and prepares for the next step.",
        "transitions": {},
    },
    "RandomnessTransactionSubmissionRound": {
        "name": "Generating randomness",
        "description": "The agent collects the randomness it needs to vary its behavior or decisions.",
        "transitions": {},
    },
    "RegistrationRound": {
        "name": "Registering agents",
        "description": "Sets up the necessary components the agent needs to operate.",
        "transitions": {},
    },
    "RegistrationStartupRound": {
        "name": "Startup registration",
        "description": "Completes setup tasks required when the agent starts running.",
        "transitions": {},
    },
    "ResetAndPauseRound": {
        "name": "Preparing for next cycle",
        "description": "The agent cleans up and pauses briefly before starting a new trading cycle.",
        "transitions": {},
    },
    "ResetRound": {
        "name": "Cleaning up and resetting",
        "description": "The agent clears temporary data and prepares for the next step.",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionARound": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBAfterTimeoutRound": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBRound": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
        "transitions": {},
    },
    "SynchronizeLateMessagesRound": {
        "name": "Syncing messages",
        "description": "The agent ensures all updates and confirmations are in sync.",
        "transitions": {},
    },
    "ValidateTransactionRound": {
        "name": "Validating the transaction",
        "description": "Checks whether the submitted transaction was completed successfully.",
        "transitions": {},
    },
    "WithdrawFundsRound": {
        "name": "Withdrawing funds",
        "description": "The agent withdraws funds back to your wallet.",
        "transitions": {},
    },
}
