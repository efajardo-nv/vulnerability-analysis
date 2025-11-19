# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import time
import typing
from contextlib import asynccontextmanager

import aiohttp

from vuln_analysis.utils.error_utils import API_KEY_SETUP_LINK
from vuln_analysis.utils.error_utils import is_authentication_error

logger = logging.getLogger(__name__)


@asynccontextmanager
async def request_with_retry(session: aiohttp.ClientSession,
                             request_kwargs: dict,
                             max_retries: int = 10,
                             sleep_time: float = 0.1,
                             respect_retry_after_header: bool = True,
                             log_on_error=True,
                             retry_on_client_errors=True,
                             api_key_env_var: str | None = None) -> typing.AsyncIterator[aiohttp.ClientResponse]:
    """
    Async version of `http_utils.request_with_retry`

    Parameters
    ----------
    api_key_env_var : str | None
        Optional name of the environment variable containing the API key.
        Used to provide better error messages for authentication failures.
    """
    assert not request_kwargs.get('raise_for_status'), "raise_for_status is incompatible with `request_with_retry`"
    try_count = 0
    done = False
    while try_count <= max_retries and not done:
        response = None
        response_headers = {}
        try:
            async with session.request(**request_kwargs) as response:
                response_headers = response.headers
                response.raise_for_status()
                yield response
                done = True
        except Exception as e:
            try_count += 1

            # Check if this is an authentication error
            is_auth_error = is_authentication_error(e)

            # Also check response headers for authentication error indicators (some APIs use non-standard codes)
            if not is_auth_error and response_headers:
                message_header = response_headers.get('message', '').lower()
                auth_keywords = ['invalid apikey', 'invalid api key', 'apikey', 'api_key',
                               'unauthorized', 'authentication failed', 'invalid key',
                               'invalid token', 'invalid credentials']
                if any(keyword in message_header for keyword in auth_keywords):
                    is_auth_error = True

            if try_count >= max_retries:
                if (log_on_error):
                    if is_auth_error and api_key_env_var:
                        logger.error(
                            "Authentication failed for %s after %d retries. "
                            "Please verify that the %s environment variable is set correctly and contains a valid API key. "
                            "See %s for setup instructions. Error: %s",
                            request_kwargs['url'],
                            max_retries,
                            api_key_env_var,
                            API_KEY_SETUP_LINK,
                            e)
                    elif is_auth_error:
                        logger.error(
                            "Authentication failed for %s after %d retries. "
                            "Please verify that your API key is valid. See %s for setup instructions. Error: %s",
                            request_kwargs['url'],
                            max_retries,
                            API_KEY_SETUP_LINK,
                            e)
                    else:
                        logger.error("Failed requesting %s after %d retries: %s", request_kwargs['url'], max_retries, e)
                raise e

            actual_sleep_time = (2**(try_count - 1)) * sleep_time

            if respect_retry_after_header and 'Retry-After' in response_headers:
                actual_sleep_time = max(int(response_headers["Retry-After"]), actual_sleep_time)
            elif respect_retry_after_header and 'X-RateLimit-Reset' in response_headers:
                actual_sleep_time = max(int(response_headers["X-RateLimit-Reset"]) - time.time(), actual_sleep_time)
            elif not retry_on_client_errors and response.status < 500:
                raise e

            if is_auth_error and api_key_env_var:
                logger.warning("Authentication error [%d/%d]: %s - Check %s environment variable. See %s (Retry %.1f sec)",
                               try_count,
                               max_retries,
                               request_kwargs['url'],
                               api_key_env_var,
                               API_KEY_SETUP_LINK,
                               actual_sleep_time)
            elif is_auth_error:
                logger.warning("Authentication error [%d/%d]: %s - Check API key. See %s (Retry %.1f sec)",
                               try_count,
                               max_retries,
                               request_kwargs['url'],
                               API_KEY_SETUP_LINK,
                               actual_sleep_time)
            else:
                logger.warning("Error requesting [%d/%d]: (Retry %.1f sec) %s: %s",
                               try_count,
                               max_retries,
                               actual_sleep_time,
                               request_kwargs['url'],
                               e)

            await asyncio.sleep(actual_sleep_time)


_T = typing.TypeVar('_T')
_P = typing.ParamSpec('_P')


def retry_async(exception_types: type[BaseException] | tuple[type[BaseException], ...] = Exception):
    """
    Retries an async function with exponential backoff

    Parameters
    ----------
    exception_types : type[BaseException] | tuple[type[BaseException], ...], optional
        The types of exceptions to trigger a retry, by default Exception
    """
    import tenacity

    def inner(func: typing.Callable[_P, typing.Awaitable[_T]]) -> typing.Callable[_P, typing.Awaitable[_T]]:

        @tenacity.retry(wait=tenacity.wait_exponential_jitter(0.1),
                        stop=tenacity.stop_after_attempt(10),
                        retry=tenacity.retry_if_exception_type(exception_types),
                        reraise=True)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return await func(*args, **kwargs)

        return wrapper

    return inner
