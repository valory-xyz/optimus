# conftest.py
import pytest
from web3 import Web3

@pytest.fixture(scope="session")
def w3():
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    assert w3.is_connected(), "Cannot connect to local Anvil node"
    return w3

@pytest.fixture(autouse=True)
def snapshot(w3):
    snap = w3.provider.make_request("evm_snapshot", [])
    yield
    w3.provider.make_request("evm_revert", [snap["result"]])
