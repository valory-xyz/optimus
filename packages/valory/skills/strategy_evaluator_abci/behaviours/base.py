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

"""This module contains the base behaviour for the 'strategy_evaluator_abci' skill."""

import json
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Sized, Tuple, cast

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload
from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseBehaviour
from packages.valory.skills.abstract_round_abci.io_.load import CustomLoaderType
from packages.valory.skills.abstract_round_abci.io_.store import (
    SupportedFiletype,
    SupportedObjectType,
)
from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.strategy_evaluator_abci.models import (
    SharedState,
    StrategyEvaluatorParams,
)
from packages.valory.skills.strategy_evaluator_abci.payloads import IPFSHashPayload
from packages.valory.skills.strategy_evaluator_abci.states.base import SynchronizedData


SWAP_DECISION_FILENAME = "swap_decision.json"
SWAP_INSTRUCTIONS_FILENAME = "swap_instructions.json"
STRATEGY_KEY = "trading_strategy"
CALLABLE_KEY = "callable"
ENTRY_POINT_STORE_KEY = "entry_point"
SUPPORTED_STRATEGY_LOG_LEVELS = ("info", "warning", "error")


def wei_to_native(wei: int) -> float:
    """Convert WEI to native token."""
    return wei / 10**18


def to_content(content: dict) -> bytes:
    """Convert the given content to bytes' payload."""
    return json.dumps(content, sort_keys=True).encode()


class StrategyEvaluatorBaseBehaviour(BaseBehaviour, ABC):
    """Represents the base class for the strategy evaluation FSM behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the strategy evaluator behaviour."""
        super().__init__(**kwargs)
        self.swap_decision_filepath = (
            Path(self.context.data_dir) / SWAP_DECISION_FILENAME
        )
        self.swap_instructions_filepath = (
            Path(self.context.data_dir) / SWAP_INSTRUCTIONS_FILENAME
        )
        self.token_balance = 0
        self.wallet_balance = 0

    @property
    def params(self) -> StrategyEvaluatorParams:
        """Return the params."""
        return cast(StrategyEvaluatorParams, self.context.params)

    @property
    def shared_state(self) -> SharedState:
        """Get the shared state."""
        return cast(SharedState, self.context.state)

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(super().synchronized_data.db)

    def strategy_store(self, strategy_name: str) -> Dict[str, str]:
        """Get the stored strategy's files."""
        return self.context.shared_state.get(strategy_name, {})

    def execute_strategy_callable(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Any] | None:
        """Execute a strategy's method and return the results."""
        trading_strategy: Optional[str] = kwargs.pop(STRATEGY_KEY, None)
        if trading_strategy is None:
            self.context.logger.error(f"No {STRATEGY_KEY!r} was given!")
            return None

        callable_key: Optional[str] = kwargs.pop(CALLABLE_KEY, None)
        if callable_key is None:
            self.context.logger.error(f"No {CALLABLE_KEY!r} was given!")
            return None

        store = self.strategy_store(trading_strategy)
        strategy_exec = store.get(ENTRY_POINT_STORE_KEY, None)
        if strategy_exec is None:
            self.context.logger.error(
                f"No executable was found for {trading_strategy=}! Did the IPFS package downloader load it correctly?"
            )
            return None

        callable_method = store.get(callable_key, None)
        if callable_method is None:
            self.context.logger.error(
                f"No {callable_method=} was found in the loaded component! "
                "Did the IPFS package downloader load it correctly?"
            )
            return None

        if callable_method in globals():
            del globals()[callable_method]

        exec(strategy_exec, globals())  # pylint: disable=W0122  # nosec
        method: Optional[Callable] = globals().get(callable_method, None)
        if method is None:
            self.context.logger.error(
                f"No {callable_method!r} method was found in {trading_strategy} strategy's executable:\n"
                f"{strategy_exec}."
            )
            return None
        # TODO this method is blocking, needs to be run from an aea skill or a task.
        return method(*args, **kwargs)

    def log_from_strategy_results(self, results: Dict[str, Any]) -> None:
        """Log any messages from a strategy's results."""
        for level in SUPPORTED_STRATEGY_LOG_LEVELS:
            logger = getattr(self.context.logger, level, None)
            if logger is not None:
                for log in results.get(level, []):
                    logger(log)

    def _handle_response(
        self,
        api: ApiSpecs,
        res: Optional[dict],
    ) -> Generator[None, None, Optional[Any]]:
        """Handle the response from an API.

        :param api: the `ApiSpecs` instance of the API.
        :param res: the response to handle.
        :return: the response's result, using the given keys. `None` if response is `None` (has failed).
        :yield: None
        """
        if res is None:
            error = f"Could not get a response from {api.api_id!r} API."
            self.context.logger.error(error)
            api.increment_retries()
            yield from self.sleep(api.retries_info.suggested_sleep_time)
            return None

        self.context.logger.info(
            f"Retrieved a response from {api.api_id!r} API: {res}."
        )
        api.reset_retries()
        return res

    def _get_response(
        self,
        api: ApiSpecs,
        dynamic_parameters: Dict[str, str],
        content: Optional[dict] = None,
    ) -> Generator[None, None, Any]:
        """Get the response from an API."""
        specs = api.get_spec()
        specs["parameters"].update(dynamic_parameters)
        if content is not None:
            specs["content"] = to_content(content)

        while not api.is_retries_exceeded():
            res_raw = yield from self.get_http_response(**specs)
            res = api.process_response(res_raw)
            response = yield from self._handle_response(api, res)
            if response is not None:
                return response

        error = f"Retries were exceeded for {api.api_id!r} API."
        self.context.logger.error(error)
        api.reset_retries()
        return None

    def get_from_ipfs(
        self,
        ipfs_hash: Optional[str],
        filetype: Optional[SupportedFiletype] = None,
        custom_loader: CustomLoaderType = None,
        timeout: Optional[float] = None,
    ) -> Generator[None, None, Optional[SupportedObjectType]]:
        """
        Gets an object from IPFS.

        If the result is `None`, then an error is logged, sleeps, and retries.

        :param ipfs_hash: the ipfs hash of the file/dir to download.
        :param filetype: the file type of the object being downloaded.
        :param custom_loader: a custom deserializer for the object received from IPFS.
        :param timeout: timeout for the request.
        :yields: None.
        :returns: the downloaded object, corresponding to ipfs_hash or `None` if retries were exceeded.
        """
        if ipfs_hash is None:
            return None

        n_retries = 0
        while n_retries < self.params.ipfs_fetch_retries:
            res = yield from super().get_from_ipfs(
                ipfs_hash, filetype, custom_loader, timeout
            )
            if res is not None:
                return res

            n_retries += 1
            sleep_time = self.params.sleep_time
            self.context.logger.error(
                f"Could not get any data from IPFS using hash {ipfs_hash!r}!"
                f"Retrying in {sleep_time}..."
            )
            yield from self.sleep(sleep_time)

        return None

    def get_ipfs_hash_payload_content(
        self,
        data: Any,
        process_fn: Callable[[Any], Generator[None, None, Tuple[Sized, bool]]],
        store_filepath: str,
    ) -> Generator[None, None, Tuple[Optional[str], Optional[bool]]]:
        """Get the ipfs hash payload's content."""
        if data is None:
            return None, None

        incomplete: Optional[bool]
        processed, incomplete = yield from process_fn(data)
        if len(processed) == 0:
            processed_hash = None
            if incomplete:
                incomplete = None
        else:
            processed_hash = yield from self.send_to_ipfs(
                store_filepath,
                processed,
                filetype=SupportedFiletype.JSON,
            )
        return processed_hash, incomplete

    def get_process_store_act(
        self,
        hash_: Optional[str],
        process_fn: Callable[[Any], Generator[None, None, Tuple[Sized, bool]]],
        store_filepath: str,
    ) -> Generator:
        """An async act method for getting some data, processing them, and storing the result.

        1. Get some data using the given hash.
        2. Process them using the given fn.
        3. Send them to IPFS using the given filepath as intermediate storage.

        :param hash_: the hash of the data to process.
        :param process_fn: the function to process the data.
        :param store_filepath: path to the file to store the processed data.
        :yield: None
        """
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            data = yield from self.get_from_ipfs(hash_, SupportedFiletype.JSON)
            sender = self.context.agent_address
            payload_data = yield from self.get_ipfs_hash_payload_content(
                data, process_fn, store_filepath
            )
            payload = IPFSHashPayload(sender, *payload_data)

        yield from self.finish_behaviour(payload)

    def finish_behaviour(self, payload: BaseTxPayload) -> Generator:
        """Finish the behaviour."""
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()
