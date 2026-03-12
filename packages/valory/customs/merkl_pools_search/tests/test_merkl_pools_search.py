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

"""Tests for merkl_pools_search custom component."""

import json
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.customs.merkl_pools_search.merkl_pools_search import (
    CAMPAIGN_TYPES,
    HTTP_OK,
    REQUIRED_FIELDS,
    DexTypes,
    check_missing_fields,
    highest_apr_opportunity,
    remove_irrelevant_fields,
    run,
)


class TestCheckMissingFields:
    """Tests for the check_missing_fields function."""

    def test_no_missing_fields(self):
        """Test that no missing fields are returned when all fields are present."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        result = check_missing_fields(kwargs)
        assert result == []

    def test_all_missing_fields(self):
        """Test that all fields are returned when none are present."""
        result = check_missing_fields({})
        assert set(result) == set(REQUIRED_FIELDS)

    def test_some_missing_fields(self):
        """Test that only missing fields are returned."""
        kwargs = {"chains": ["optimism"], "apr_threshold": 5.0}
        result = check_missing_fields(kwargs)
        assert "chains" not in result
        assert "apr_threshold" not in result
        for field in REQUIRED_FIELDS:
            if field not in kwargs:
                assert field in result

    def test_none_value_counts_as_missing(self):
        """Test that fields with None value count as missing."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        kwargs["chains"] = None
        result = check_missing_fields(kwargs)
        assert "chains" in result


class TestRemoveIrrelevantFields:
    """Tests for the remove_irrelevant_fields function."""

    def test_removes_irrelevant_fields(self):
        """Test that irrelevant fields are removed."""
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        kwargs["extra_field"] = "should_be_removed"
        result = remove_irrelevant_fields(kwargs)
        assert "extra_field" not in result
        for field in REQUIRED_FIELDS:
            assert field in result

    def test_empty_kwargs(self):
        """Test with empty kwargs."""
        result = remove_irrelevant_fields({})
        assert result == {}

    def test_only_relevant_fields(self):
        """Test with only relevant fields."""
        kwargs = {field: f"val_{i}" for i, field in enumerate(REQUIRED_FIELDS)}
        result = remove_irrelevant_fields(kwargs)
        assert result == kwargs


class TestHighestAprOpportunity:
    """Tests for the highest_apr_opportunity function."""

    def _base_params(self):
        """Return base parameters for highest_apr_opportunity."""
        return {
            "balancer_graphql_endpoints": {
                "optimism": "https://api.example.com/graphql"
            },
            "merkl_fetch_campaign_args": {
                "url": "https://api.merkl.xyz/v4/campaigns",
                "creator": "test_creator",
                "live": "true",
            },
            "chains": ["optimism"],
            "protocols": ["balancer"],
            "chain_to_chain_id_mapping": {"optimism": 10},
            "apr_threshold": 5.0,
            "current_pool": "0xcurrentpool",
        }

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_empty_chains(self, mock_get):
        """Test with empty chains list."""
        params = self._base_params()
        params["chains"] = []
        result = highest_apr_opportunity(**params)
        assert "error" in result
        assert "No chain selected" in result["error"]

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_api_error_status_code(self, mock_get):
        """Test when API returns error status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result
        assert "Status code 500" in result["error"]

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_api_json_parse_error(self, mock_get):
        """Test when API returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "not valid json {"
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result
        assert "Could not parse response" in result["error"]

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_no_eligible_pools(self, mock_get):
        """Test when no pools match criteria."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "10": {
                    "campaign1": {
                        "c1": {
                            "type": "SomeOtherType",
                            "apr": 10,
                            "mainParameter": "0xpool",
                        }
                    }
                }
            }
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_pool_with_zero_apr(self, mock_get):
        """Test that pools with zero APR are filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "10": {
                    "campaign1": {
                        "c1": {
                            "type": "balancerPool",
                            "apr": 0,
                            "mainParameter": "0xpool",
                        }
                    }
                }
            }
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_pool_with_negative_apr(self, mock_get):
        """Test that pools with negative APR are filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "10": {
                    "campaign1": {
                        "c1": {
                            "type": "balancerPool",
                            "apr": -5,
                            "mainParameter": "0xpool",
                        }
                    }
                }
            }
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_pool_below_apr_threshold(self, mock_get):
        """Test that pools below APR threshold are filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "10": {
                    "campaign1": {
                        "c1": {
                            "type": "balancerPool",
                            "apr": 3.0,
                            "mainParameter": "0xpool",
                        }
                    }
                }
            }
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_pool_is_current_pool(self, mock_get):
        """Test that the current pool is filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "10": {
                    "campaign1": {
                        "c1": {
                            "type": "balancerPool",
                            "apr": 50.0,
                            "mainParameter": "0xcurrentpool",
                        }
                    }
                }
            }
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_pool_with_no_main_parameter(self, mock_get):
        """Test that pools without mainParameter are filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {"10": {"campaign1": {"c1": {"type": "balancerPool", "apr": 50.0}}}}
        )
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_balancer_pool_successful(self, mock_get, mock_post):
        """Test successful Balancer pool processing with valid tokensList of length 2."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "ammName": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        # Mock graphql responses for pool type and tokensList
        mock_pool_type_response = MagicMock()
        mock_pool_type_response.status_code = 200
        mock_pool_type_response.text = json.dumps(
            {"data": {"pools": [{"id": "0xpoolid", "poolType": "Weighted"}]}}
        )
        mock_tokens_list_response = MagicMock()
        mock_tokens_list_response.status_code = 200
        mock_tokens_list_response.text = json.dumps(
            {
                "data": {
                    "pools": [
                        {"id": "0xpoolid", "tokensList": ["0xtoken0", "0xtoken1"]}
                    ]
                }
            }
        )
        mock_post.side_effect = [mock_pool_type_response, mock_tokens_list_response]

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" not in result
        assert result["dex_type"] == "balancerPool"
        assert result["pool_address"] == "0xpooladdress"
        assert result["apr"] == 50.0

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_balancer_pool_type_none_skips(self, mock_get, mock_post):
        """Test that Balancer pool with None pool_type is skipped."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        # Return no pools from graphql (so poolType is None)
        mock_graphql_response = MagicMock()
        mock_graphql_response.status_code = 200
        mock_graphql_response.text = json.dumps({"data": {"pools": []}})
        mock_post.return_value = mock_graphql_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_balancer_pool_wrong_token_count(self, mock_get, mock_post):
        """Test that Balancer pool with != 2 tokens is skipped."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        # Pool type ok but 3 tokens
        mock_pool_type = MagicMock()
        mock_pool_type.status_code = 200
        mock_pool_type.text = json.dumps(
            {"data": {"pools": [{"id": "0xpoolid", "poolType": "Weighted"}]}}
        )
        mock_tokens_list = MagicMock()
        mock_tokens_list.status_code = 200
        mock_tokens_list.text = json.dumps(
            {
                "data": {
                    "pools": [
                        {"id": "0xpoolid", "tokensList": ["0xt0", "0xt1", "0xt2"]}
                    ]
                }
            }
        )
        mock_post.side_effect = [mock_pool_type, mock_tokens_list]

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_uniswap_v3_pool_successful(self, mock_get):
        """Test successful UniswapV3 pool processing."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "UniswapV3",
                        "apr": 40.0,
                        "mainParameter": "0xunipooladdress",
                        "campaignParameters": {
                            "token0": "0xtoken0",
                            "token1": "0xtoken1",
                            "symbolToken0": "TOKEN0",
                            "symbolToken1": "TOKEN1",
                            "poolFee": 3000,
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        params["protocols"] = ["uniswap_v3"]
        result = highest_apr_opportunity(**params)
        assert "error" not in result
        assert result["dex_type"] == "UniswapV3"
        assert result["pool_address"] == "0xunipooladdress"
        assert result["pool_fee"] == 3000

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_uniswap_v3_pool_missing_campaign_parameters(self, mock_get):
        """Test UniswapV3 pool with empty campaignParameters."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "UniswapV3",
                        "apr": 40.0,
                        "mainParameter": "0xunipooladdress",
                        "campaignParameters": {},
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        params["protocols"] = ["uniswap_v3"]
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_uniswap_v3_pool_none_token_values(self, mock_get):
        """Test UniswapV3 pool with None in token values."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "UniswapV3",
                        "apr": 40.0,
                        "mainParameter": "0xunipooladdress",
                        "campaignParameters": {
                            "token0": None,
                            "token1": "0xtoken1",
                            "symbolToken0": "TOKEN0",
                            "symbolToken1": "TOKEN1",
                            "poolFee": 3000,
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        params["protocols"] = ["uniswap_v3"]
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_balancer_pool_insufficient_tokens(self, mock_get, mock_post):
        """Test Balancer pool with fewer than 2 pool tokens."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                            },
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_balancer_pool_token_missing_symbol(self, mock_get, mock_post):
        """Test Balancer pool where a token is missing its symbol."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": None},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_no_endpoint(self, mock_get, mock_post):
        """Test fetch_balancer_pool_info when no graphql endpoint exists for chain."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        # Remove the graphql endpoint for the chain
        params["balancer_graphql_endpoints"] = {}
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_api_error(self, mock_get, mock_post):
        """Test fetch_balancer_pool_info when GraphQL API returns error status."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        mock_graphql_response = MagicMock()
        mock_graphql_response.status_code = 500
        mock_post.return_value = mock_graphql_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_json_decode_error(self, mock_get, mock_post):
        """Test fetch_balancer_pool_info when GraphQL returns invalid JSON."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        mock_graphql_response = MagicMock()
        mock_graphql_response.status_code = 200
        mock_graphql_response.text = "not json"
        mock_post.return_value = mock_graphql_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_null_response(self, mock_get, mock_post):
        """Test fetch_balancer_pool_info when GraphQL returns null."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        mock_graphql_response = MagicMock()
        mock_graphql_response.status_code = 200
        mock_graphql_response.text = "null"
        mock_post.return_value = mock_graphql_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_no_pools_in_response(self, mock_get, mock_post):
        """Test fetch_balancer_pool_info when GraphQL response has empty pools array."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        mock_graphql_response = MagicMock()
        mock_graphql_response.status_code = 200
        mock_graphql_response.text = json.dumps({"data": {"pools": []}})
        mock_post.return_value = mock_graphql_response

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.post"
    )
    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_fetch_balancer_pool_info_tokens_error(self, mock_get, mock_post):
        """Test when tokensList fetch returns an error."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": "balancerPool",
                        "apr": 50.0,
                        "mainParameter": "0xpooladdress",
                        "typeInfo": {
                            "poolId": "0xpoolid",
                            "poolTokens": {
                                "0xtoken0": {"symbol": "TOKEN0"},
                                "0xtoken1": {"symbol": "TOKEN1"},
                            },
                        },
                    }
                }
            }
        }
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_api_response

        # First call pool type ok, second call tokens returns error
        mock_pool_type = MagicMock()
        mock_pool_type.status_code = 200
        mock_pool_type.text = json.dumps(
            {"data": {"pools": [{"id": "0xpoolid", "poolType": "Weighted"}]}}
        )
        mock_tokens_error = MagicMock()
        mock_tokens_error.status_code = 500
        mock_post.side_effect = [mock_pool_type, mock_tokens_error]

        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_dex_type_from_ammname_fallback(self, mock_get):
        """Test that pools use ammName when type field is None."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "type": None,
                        "ammName": "UniswapV3",
                        "apr": 40.0,
                        "mainParameter": "0xunipooladdress",
                        "campaignParameters": {
                            "token0": "0xtoken0",
                            "token1": "0xtoken1",
                            "symbolToken0": "TOKEN0",
                            "symbolToken1": "TOKEN1",
                            "poolFee": 3000,
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        params["protocols"] = ["uniswap_v3"]
        result = highest_apr_opportunity(**params)
        assert "error" not in result
        assert result["dex_type"] == "UniswapV3"

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_dex_type_from_ammname_field(self, mock_get):
        """Test that dex_type can come from ammName field when type is missing."""
        campaign_data = {
            "10": {
                "campaign_group": {
                    "campaign_1": {
                        "ammName": "UniswapV3",
                        "apr": 40.0,
                        "mainParameter": "0xunipooladdress",
                        "campaignParameters": {
                            "token0": "0xtoken0",
                            "token1": "0xtoken1",
                            "symbolToken0": "TOKEN0",
                            "symbolToken1": "TOKEN1",
                            "poolFee": 3000,
                        },
                    }
                }
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(campaign_data)
        mock_get.return_value = mock_response

        params = self._base_params()
        params["protocols"] = ["uniswap_v3"]
        result = highest_apr_opportunity(**params)
        assert "error" not in result
        assert result["dex_type"] == "UniswapV3"

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_live_default_value(self, mock_get):
        """Test that 'live' parameter defaults to 'true'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({})
        mock_get.return_value = mock_response

        params = self._base_params()
        del params["merkl_fetch_campaign_args"]["live"]
        highest_apr_opportunity(**params)
        call_url = mock_get.call_args[0][0]
        assert "live=true" in call_url

    @patch("packages.valory.customs.merkl_pools_search.merkl_pools_search.requests.get")
    def test_api_201_status_accepted(self, mock_get):
        """Test that HTTP 201 status code is accepted."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = json.dumps({})
        mock_get.return_value = mock_response
        params = self._base_params()
        result = highest_apr_opportunity(**params)
        assert "error" in result  # No pools but no HTTP error


class TestRun:
    """Tests for the run function."""

    def test_missing_required_fields(self):
        """Test run with missing required fields."""
        result = run()
        assert "error" in result
        assert "Required kwargs" in result["error"]

    def test_partial_missing_fields(self):
        """Test run with some missing fields."""
        result = run(chains=["optimism"])
        assert "error" in result

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.highest_apr_opportunity"
    )
    def test_run_delegates_to_highest_apr(self, mock_func):
        """Test that run properly delegates to highest_apr_opportunity."""
        mock_func.return_value = {"pool_address": "0x123"}
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        result = run(**kwargs)
        mock_func.assert_called_once()
        assert result == {"pool_address": "0x123"}

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.highest_apr_opportunity"
    )
    def test_run_strips_irrelevant_fields(self, mock_func):
        """Test that run strips extra kwargs before delegating."""
        mock_func.return_value = {}
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        kwargs["extra"] = "should_be_removed"
        run(**kwargs)
        call_kwargs = mock_func.call_args[1]
        assert "extra" not in call_kwargs

    @patch(
        "packages.valory.customs.merkl_pools_search.merkl_pools_search.highest_apr_opportunity"
    )
    def test_run_with_positional_args(self, mock_func):
        """Test that run ignores positional arguments."""
        mock_func.return_value = {}
        kwargs = {field: "value" for field in REQUIRED_FIELDS}
        run("some_positional_arg", **kwargs)
        mock_func.assert_called_once()
