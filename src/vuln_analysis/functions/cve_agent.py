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

import asyncio
import contextvars
import logging
import typing
import warnings
from typing import Any

from aiolimiter import AsyncLimiter
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain.agents.agent import RunnableAgent
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field
from pydantic import model_validator

from vuln_analysis.data_models.state import AgentMorpheusEngineState
from vuln_analysis.utils.concurrency import get_effective_rate_limiter
from vuln_analysis.utils.prompting import get_agent_prompt

logger = logging.getLogger(__name__)

# Context variables shared between agent and its tools:
# - ctx_state: Provides tools access to workflow state (VDB paths, code index paths, CVE intel)
# - ctx_rate_limiter: Enables tools to use same LLM rate limit as the agent
ctx_state = contextvars.ContextVar("ctx_state", default="default_value")
ctx_rate_limiter = contextvars.ContextVar("ctx_rate_limiter", default=None)


class RateLimitingCallback(AsyncCallbackHandler):
    """
    LangChain callback handler that applies rate limiting to LLM calls.
    This ensures that the rate limit is respected even when the LLM is called multiple times
    by an agent during its execution loop.
    """

    def __init__(self, rate_limiter: AsyncLimiter | None):
        super().__init__()
        self.rate_limiter = rate_limiter

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Apply rate limiting before each LLM call starts."""
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()


class CVEAgentExecutorToolConfig(FunctionBaseConfig, name="cve_agent_executor"):
    """
    Defines a function that iterates through checklist items using provided tools and gathered intel.
    """
    llm_name: str = Field(description="The LLM model to use with the CVE agent.")
    tool_names: list[str] = Field(default=[], description="The list of tools to provide to CVE agent.")
    llm_max_rate: int | None = Field(
        default=None,
        description="Maximum LLM rate limit (requests per second) for all LLM calls made by the agent or its tools. "
        "If set to a number, overrides the workflow-level llm_max_rate. If None, inherits from workflow-level setting. "
        "Takes precedence over the deprecated max_concurrency parameter if both are specified.")
    max_concurrency: int | None = Field(
        default=None,
        description="DEPRECATED: Use llm_max_rate instead. "
        "This parameter is now treated as llm_max_rate (requests/second) for rate limiting, "
        "not as a concurrent operation limit. This behavior change may affect performance. "
        "If both max_concurrency and llm_max_rate are specified, llm_max_rate takes precedence. "
        "This parameter will be removed in a future version.")
    max_iterations: int = Field(default=10, description="The maximum number of iterations for the agent.")
    prompt: str | None = Field(
        default=None,
        description=
        "Manually set the prompt for the specific model in the configuration. The prompt can either be passed in as a "
        "string of text or as a path to a text file containing the desired prompting.")
    prompt_examples: bool = Field(default=False, description="Whether to include examples in agent prompt.")
    replace_exceptions: bool = Field(default=False,
                                     description="Whether to replace exception message with custom message.")
    replace_exceptions_value: str | None = Field(default=None, description="Message if replace_exceptions is true")
    return_intermediate_steps: bool = Field(
        default=False,
        description=
        "Controls whether to return intermediate steps taken by the agent, and include them in the output file.")
    verbose: bool = Field(default=False, description="Set to true for verbose output")

    @model_validator(mode="after")
    def validate_max_concurrency_deprecated(self):
        """Warn if max_concurrency is used and suggest llm_max_rate instead."""
        if self.max_concurrency is not None:
            if self.llm_max_rate is not None:
                # Both parameters specified - llm_max_rate takes precedence
                msg = (
                    f"The 'max_concurrency' parameter is deprecated and will be removed in a future version. "
                    f"Both max_concurrency={self.max_concurrency} and llm_max_rate={self.llm_max_rate} are specified. "
                    f"Using llm_max_rate={self.llm_max_rate} (max_concurrency will be ignored). "
                    f"Please remove max_concurrency from your configuration.")
            else:
                # Only max_concurrency specified
                msg = ("The 'max_concurrency' parameter is deprecated and will be removed in a future version. "
                       "Please use 'llm_max_rate' (requests per second) instead for better rate limiting control.")

            # Use logging for guaranteed visibility
            logger.warning(msg)

            # Also emit proper deprecation warning for tooling/static analysis
            warnings.warn(msg, DeprecationWarning, stacklevel=1)
        return self


async def _create_agent(config: CVEAgentExecutorToolConfig,
                        builder: Builder,
                        state: AgentMorpheusEngineState,
                        rate_limiter: AsyncLimiter | None) -> tuple[AgentExecutor, list]:
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt = PromptTemplate.from_template(get_agent_prompt(config.prompt, config.prompt_examples))

    # Create rate limiting callback to control all LLM calls made by the agent
    callbacks = [RateLimitingCallback(rate_limiter)] if rate_limiter is not None else []

    # Filter tools that are not available
    tools = [
        tool for tool in tools
        if not ((tool.name == "Container Image Code QA System" and state.code_vdb_path is None) or
                (tool.name == "Container Image Developer Guide QA System" and state.doc_vdb_path is None) or
                (tool.name == "Lexical Search Container Image Code QA System" and state.code_index_path is None))
    ]

    agent = create_react_agent(llm=llm,
                               tools=tools,
                               prompt=prompt,
                               stop_sequence=["\nObservation:", "\n\tObservation:"])

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        early_stopping_method="force",
        handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
        max_iterations=config.max_iterations,
        return_intermediate_steps=config.return_intermediate_steps,
        verbose=config.verbose)

    # Disable streaming for accurate token counts
    if isinstance(agent_executor.agent, RunnableAgent):
        agent_executor.agent.stream_runnable = False

    return agent_executor, callbacks


async def _process_steps(agent, steps, callbacks):

    async def _process_step(step):
        try:
            return await agent.ainvoke({"input": step}, config={"callbacks": callbacks})
        except Exception as e:
            logger.error("Error in agent execution: %s: %s", type(e).__name__, e)
            raise

    return await asyncio.gather(*(_process_step(step) for step in steps))


def _parse_intermediate_step(step: tuple[typing.Any, typing.Any]) -> dict[str, typing.Any]:
    """
    Parse an agent intermediate step into an AgentIntermediateStep object. Return the dictionary representation for
    compatibility with cudf.
    """
    if len(step) != 2:
        raise ValueError(f"Expected 2 values in each intermediate step but got {len(step)}.")

    action, output = step

    return {"tool_name": action.tool, "action_log": action.log, "tool_input": action.tool_input, "tool_output": output}


def _postprocess_results(results: list[list[dict]], replace_exceptions: bool,
                         replace_exceptions_value: str | None) -> tuple[list[list[str]], list[list[list]]]:
    """
    Post-process results into lists of outputs and intermediate steps. Replace exceptions with placholder values if
    config.replace_exceptions = True.
    """

    for i, answer_list in enumerate(results):
        for j, answer in enumerate(answer_list):

            # Handle exceptions returned by the agent
            # OutputParserException is not a subclass of Exception, so we need to check for it separately
            if isinstance(answer, (OutputParserException, Exception)):
                if replace_exceptions:
                    # If the agent encounters a parsing error or a server error after retries, replace the error
                    # with default values to prevent the pipeline from crashing
                    results[i][j]["output"] = replace_exceptions_value
                    results[i][j]["intermediate_step"] = None
                    logger.warning(
                        "Error in agent execution for result[%d][%d]: %s. "
                        "Replacing with default output: \"%s\" and intermediate_steps: None",
                        i,
                        j,
                        answer,
                        replace_exceptions_value)

            # For successful agent responses, extract the output, and intermediate steps if available
            else:
                # intermediate_steps availability depends on config.return_intermediate_steps
                if "intermediate_steps" in answer:
                    results[i][j]["intermediate_steps"] = [
                        _parse_intermediate_step(step) for step in answer["intermediate_steps"]
                    ]
                else:
                    results[i][j]["intermediate_steps"] = None

    return results


@register_function(config_type=CVEAgentExecutorToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def cve_agent(config: CVEAgentExecutorToolConfig, builder: Builder):

    async def _arun(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        ctx_state.set(state)

        # Handle deprecated max_concurrency parameter
        effective_max_rate = config.llm_max_rate
        if effective_max_rate is None and config.max_concurrency is not None:
            # For backward compatibility, convert max_concurrency to llm_max_rate (requests per second)
            # NOTE: This changes behavior from concurrency limiting to rate limiting
            effective_max_rate = config.max_concurrency
            logger.warning(
                "Deprecated max_concurrency=%d is now treated as llm_max_rate=%d requests/second (rate limiting). "
                "This changes from concurrent operation limiting and may affect performance. "
                "Please update your configuration to use llm_max_rate.",
                config.max_concurrency,
                effective_max_rate)

        rate_limiter = get_effective_rate_limiter(effective_max_rate, builder)

        # Store the rate limiter in context so tools can create their own callbacks
        ctx_rate_limiter.set(rate_limiter)

        # Create agent with rate-limiting callback
        agent, callbacks = await _create_agent(config, builder, state, rate_limiter)

        if rate_limiter is not None:
            effective_rate = int(rate_limiter.max_rate / rate_limiter.time_period)
            logger.info("Executing agent for %d CVEs with LLM rate limit of %d requests/second",
                        len(state.checklist_plans),
                        effective_rate)
        else:
            logger.info("Executing agent for %d CVEs with no rate limiting", len(state.checklist_plans))

        # Process all CVEs and their steps in parallel - the rate limiter callback will control the actual LLM request rate
        results = await asyncio.gather(*(_process_steps(agent, steps, callbacks)
                                         for steps in state.checklist_plans.values()))
        results = _postprocess_results(results, config.replace_exceptions, config.replace_exceptions_value)
        state.checklist_results = dict(zip(state.checklist_plans.keys(), results))
        return state

    yield FunctionInfo.from_fn(
        _arun,
        input_schema=AgentMorpheusEngineState,
        description=("Executes provided checklist of tasks mapped to flagged CVEs to investigate the "
                     "exploitability of a software container by the flagged CVEs."))
