# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Test the io_/loader.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

import pytest

from packages.valory.skills.liquidity_trader_abci.io_.loader import ComponentPackageLoader


def test_import() -> None:
    """Test that the loader module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.io_.loader  # noqa


class TestComponentPackageLoader:
    """Test ComponentPackageLoader class."""

    def test_load_success(self) -> None:
        """Test successful loading of a component package."""
        serialized_objects = {
            "component.yaml": "entry_point: main.py\ncallable: run",
            "main.py": "def run(): pass",
        }
        component_yaml, entry_point, callable_method = ComponentPackageLoader.load(
            serialized_objects
        )
        assert component_yaml["entry_point"] == "main.py"
        assert component_yaml["callable"] == "run"
        assert entry_point == "def run(): pass"
        assert callable_method == "run"

    def test_load_missing_component_yaml(self) -> None:
        """Test loading raises when component.yaml is missing."""
        serialized_objects = {
            "main.py": "def run(): pass",
        }
        with pytest.raises(ValueError, match="MUST contain a component.yaml"):
            ComponentPackageLoader.load(serialized_objects)

    def test_load_missing_entry_point_key(self) -> None:
        """Test loading raises when entry_point key is missing from component.yaml."""
        serialized_objects = {
            "component.yaml": "callable: run",
            "main.py": "def run(): pass",
        }
        with pytest.raises(ValueError, match="MUST contain the 'entry_point' and 'callable' keys"):
            ComponentPackageLoader.load(serialized_objects)

    def test_load_missing_callable_key(self) -> None:
        """Test loading raises when callable key is missing from component.yaml."""
        serialized_objects = {
            "component.yaml": "entry_point: main.py",
            "main.py": "def run(): pass",
        }
        with pytest.raises(ValueError, match="MUST contain the 'entry_point' and 'callable' keys"):
            ComponentPackageLoader.load(serialized_objects)

    def test_load_missing_entry_point_file(self) -> None:
        """Test loading raises when entry_point file is missing from package."""
        serialized_objects = {
            "component.yaml": "entry_point: main.py\ncallable: run",
        }
        with pytest.raises(ValueError, match="main.py is not present"):
            ComponentPackageLoader.load(serialized_objects)
