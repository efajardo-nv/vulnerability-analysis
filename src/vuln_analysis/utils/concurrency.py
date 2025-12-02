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
Concurrency utilities for the vulnerability analysis workflow.

This module provides rate limiting utilities to control the rate of concurrent
LLM API requests across workflow functions. Rate limiting helps prevent API
throttling errors and ensures stable performance when making multiple LLM calls.
"""

import contextvars
from typing import TYPE_CHECKING

from aiolimiter import AsyncLimiter

if TYPE_CHECKING:
    from nat.builder.builder import Builder

# Context variable to store parent workflow's llm_max_rate
# This allows nested workflows (like cve_agent used inside execute_then_select_function)
# to propagate their llm_max_rate setting to child functions
ctx_parent_max_rate: contextvars.ContextVar[int | None] = contextvars.ContextVar("ctx_parent_max_rate", default=None)


def get_effective_rate_limiter(local_max_rate: int | None, builder: "Builder") -> AsyncLimiter | None:
    """
    Resolve the effective rate limiter with strict no-burst rate limiting.

    Priority:
    1. Local function-specific llm_max_rate (if not None)
    2. Parent workflow llm_max_rate from context (if not None)
    3. Top-level workflow llm_max_rate (if not None)
    4. None (no rate limiting)

    The rate limiter is configured to prevent bursts by setting max_rate=1 and
    time_period=1/desired_rate. For example, llm_max_rate=5 (requests/second) becomes
    AsyncLimiter(max_rate=1, time_period=0.2), enforcing exactly 0.2 seconds between requests.

    Args:
        local_max_rate: The function-specific llm_max_rate value (requests per second)
        builder: The Builder instance to get workflow config from

    Returns:
        An AsyncLimiter instance configured with strict rate limiting, or None for no rate limiting
    """
    effective_max_rate = None

    if local_max_rate is not None:
        effective_max_rate = local_max_rate
    else:
        # Check for parent workflow llm_max_rate from context
        # This handles cases like cve_agent used inside execute_then_select_function
        parent_max_rate = ctx_parent_max_rate.get()
        if parent_max_rate is not None:
            effective_max_rate = parent_max_rate
        else:
            # Get workflow-level llm_max_rate from workflow config
            try:
                workflow_config = builder.get_workflow_config()
                if workflow_config:
                    workflow_max_rate = getattr(workflow_config, 'llm_max_rate', None)
                    if workflow_max_rate is not None:
                        effective_max_rate = workflow_max_rate
            except ValueError:
                # No workflow set - this can happen during function building
                pass

    # Return None if no rate limiting is configured
    if effective_max_rate is None:
        return None

    # Convert user-specified llm_max_rate to no-burst configuration e.g., instead of AsyncLimiter(max_rate=5, time_period=1)
    # which allows bursts, use AsyncLimiter(max_rate=1, time_period=1/5) to enforce strict rate limiting. This ensures
    # exactly 1/effective_max_rate seconds between each request.
    time_between_requests = 1.0 / effective_max_rate
    return AsyncLimiter(max_rate=1, time_period=time_between_requests)
