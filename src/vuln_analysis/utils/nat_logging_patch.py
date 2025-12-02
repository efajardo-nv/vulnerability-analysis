# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Patch for NAT framework logging to enhance error messages.

This module patches the NAT Function class's ainvoke method to:
1. Truncate long input values in error messages
2. Detect authentication errors and provide context about which API key to check

The truncation length can be configured via the NAT_ERROR_LOG_MAX_LENGTH environment
variable (default: 500 characters). Set to 0 to disable truncation.

Example:
    export NAT_ERROR_LOG_MAX_LENGTH=1000  # Truncate at 1000 chars
    export NAT_ERROR_LOG_MAX_LENGTH=0     # Disable truncation (show full error)
"""

import logging
import os
import re

from vuln_analysis.utils.error_utils import API_KEY_SETUP_LINK
from vuln_analysis.utils.error_utils import detect_api_key_from_error
from vuln_analysis.utils.error_utils import is_authentication_error

logger = logging.getLogger(__name__)

# Configure truncation length (characters)
# Can be overridden via NAT_ERROR_LOG_MAX_LENGTH environment variable
DEFAULT_MAX_LOG_LENGTH = 500
MAX_LOG_LENGTH = int(os.environ.get('NAT_ERROR_LOG_MAX_LENGTH', DEFAULT_MAX_LOG_LENGTH))


def _truncate_value(value, max_length=MAX_LOG_LENGTH):
    """Truncate a value for logging purposes."""
    value_str = str(value)
    # If max_length is 0, truncation is disabled
    if max_length == 0 or len(value_str) <= max_length:
        return value_str
    return f"{value_str[:max_length]}... [truncated, total length: {len(value_str)} chars]"


def _detect_auth_error_and_api_key(exception: Exception) -> tuple[bool, str | None]:
    """
    Detect if an exception is an authentication error and identify the relevant API key.

    Returns:
        tuple: (is_auth_error, api_key_env_var)
    """
    # Use shared utility to detect authentication errors
    auth_error = is_authentication_error(exception)

    if not auth_error:
        return (False, None)

    # Try to detect which API key from the error
    api_key_env_var = detect_api_key_from_error(exception)

    # If we can't determine the specific key, provide a generic message
    if api_key_env_var is None:
        api_key_env_var = 'NVIDIA_API_KEY (or OPENAI_API_KEY if using OpenAI)'

    return (True, api_key_env_var)


def apply_nat_logging_patch():
    """
    Apply a runtime patch to NAT's Function class to enhance error logging.

    This patches the ainvoke method to:
    1. Truncate long input values in error logs
    2. Detect authentication errors and provide context about which API key to check
    """
    try:
        from nat.builder.function import Function

        # Store the original ainvoke method
        original_ainvoke = Function.ainvoke

        async def patched_ainvoke(self, value, to_type=None):
            """Patched ainvoke with enhanced error logging."""
            # Use context manager exactly as in original implementation
            with self._context.push_active_function(self.instance_name, input_data=value) as manager:
                try:
                    converted_input = self._convert_input(value)
                    result = await self._ainvoke(converted_input)

                    if to_type is not None and not isinstance(result, to_type):
                        result = self.convert(result, to_type)

                    manager.set_output(result)
                    return result
                except Exception as e:
                    # Detect if this is an authentication error
                    is_auth_error, api_key_env_var = _detect_auth_error_and_api_key(e)

                    # Log with truncated input value and enhanced error context
                    truncated_value = _truncate_value(value)
                    nat_logger = logging.getLogger('nat.builder.function')

                    if is_auth_error and api_key_env_var:
                        nat_logger.error(
                            "Authentication error in function '%s'. "
                            "Please verify that the %s environment variable is set correctly and contains a valid API key. "
                            "See %s for setup instructions. Input: %s. Error: %s",
                            self.instance_name,
                            api_key_env_var,
                            API_KEY_SETUP_LINK,
                            truncated_value,
                            e)
                    else:
                        nat_logger.error("Error with ainvoke in function '%s' with input: %s. %s: %s",
                                         self.instance_name,
                                         truncated_value,
                                         type(e).__name__,
                                         e)
                    raise

        # Apply the patch
        Function.ainvoke = patched_ainvoke
        logger.debug("Successfully applied NAT logging patch with authentication error detection (max_length=%d)",
                     MAX_LOG_LENGTH)

    except ImportError as e:
        logger.warning("Could not apply NAT logging patch: NAT framework not found. Error: %s", e)
    except Exception as e:
        logger.warning("Failed to apply NAT logging patch: %s", e)
