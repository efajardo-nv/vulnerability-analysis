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
"""Utilities for error handling and detection."""

import aiohttp
import requests

# Link to API key setup instructions
API_KEY_SETUP_LINK = "https://github.com/NVIDIA-AI-Blueprints/vulnerability-analysis?tab=readme-ov-file#obtain-api-keys"


def is_authentication_error(exception: Exception) -> bool:
    """
    Check if an exception is related to authentication or authorization.

    These errors typically indicate API key issues and often don't require
    full stack traces for debugging - the user usually just needs to check
    their API key configuration.

    Handles multiple HTTP client libraries (aiohttp, requests) and checks for:
    - HTTP 401/403 status codes
    - Authentication-related keywords in error messages
    - PermissionError exceptions

    Parameters
    ----------
    exception : Exception
        The exception to check

    Returns
    -------
    bool
        True if the exception is authentication/authorization related

    Examples
    --------
    >>> try:
    ...     async with session.get(url) as resp:
    ...         resp.raise_for_status()
    ... except Exception as e:
    ...     if is_authentication_error(e):
    ...         logger.error("Auth error: %s", e)  # No stack trace needed
    ...     else:
    ...         logger.error("Unexpected error: %s", e, exc_info=True)
    """
    # Check for aiohttp ClientResponseError with 401/403 status codes
    if isinstance(exception, aiohttp.ClientResponseError):
        if exception.status in (401, 403):
            return True

    # Check for requests HTTPError with 401/403 status codes
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception.response, 'status_code'):
            if exception.response.status_code in (401, 403):
                return True

    # Check for PermissionError
    if isinstance(exception, PermissionError):
        return True

    # Check error message for authentication-related keywords
    error_str = str(exception).lower()
    auth_keywords = [
        'invalid apikey',
        'invalid api key',
        'apikey',
        'api_key',
        'unauthorized',
        'authentication failed',
        'authentication error',
        'invalid key',
        'invalid token',
        'invalid credentials',
        'forbidden',
        '401',
        '403'
    ]

    return any(keyword in error_str for keyword in auth_keywords)


def detect_api_key_from_error(exception: Exception) -> str | None:
    """
    Try to determine which API key environment variable is causing the error.

    Analyzes the exception message to identify which service's API key is likely
    the cause of an authentication error. This helps provide more specific
    error messages to users.

    Parameters
    ----------
    exception : Exception
        The exception to analyze

    Returns
    -------
    str | None
        The likely environment variable name (e.g., 'NVIDIA_API_KEY'),
        or None if it cannot be determined

    Examples
    --------
    >>> api_key = detect_api_key_from_error(e)
    >>> if api_key:
    ...     logger.error("Check %s environment variable", api_key)
    """
    error_str = str(exception).lower()

    # Check for service-specific indicators
    if 'nvidia' in error_str or 'nim' in error_str or 'integrate.api.nvidia' in error_str:
        return 'NVIDIA_API_KEY'
    elif 'openai' in error_str or 'gpt' in error_str:
        return 'OPENAI_API_KEY'
    elif 'serpapi' in error_str or 'serp' in error_str:
        return 'SERPAPI_API_KEY'
    elif 'github' in error_str or 'ghsa' in error_str:
        return 'GHSA_API_KEY'
    elif 'nvd' in error_str:
        return 'NVD_API_KEY'

    return None
