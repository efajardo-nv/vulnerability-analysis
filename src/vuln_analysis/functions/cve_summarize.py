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
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from vuln_analysis.utils.concurrency import get_effective_rate_limiter
from vuln_analysis.utils.string_utils import get_checklist_item_string

logger = logging.getLogger(__name__)


class CVESummarizeToolConfig(FunctionBaseConfig, name="cve_summarize"):
    """
    Defines a function that generates concise, human-readable summarization paragraph from agent results.
    """
    llm_name: str = Field(description="The LLM model to use")
    llm_max_rate: int | None = Field(
        default=None,
        description="Maximum LLM rate limit (requests per second) for summarization tasks. "
        "If set to a number, overrides the workflow-level llm_max_rate. If None, inherits from workflow-level setting.")


@register_function(config_type=CVESummarizeToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def cve_summarize(config: CVESummarizeToolConfig, builder: Builder):

    from langchain_core.prompts import PromptTemplate

    from vuln_analysis.data_models.state import AgentMorpheusEngineState
    from vuln_analysis.utils.prompting import SUMMARY_PROMPT

    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt = PromptTemplate(input_variables=["response"], template=SUMMARY_PROMPT)
    chain = prompt | llm

    async def summarize_cve(results):
        try:
            response = '\n'.join(
                [get_checklist_item_string(idx + 1, checklist_item) for idx, checklist_item in enumerate(results[1])])
            final_summary = await chain.ainvoke({"response": response})

            return final_summary.content
        except Exception as e:
            logger.error("Error in generating summary: %s: %s", type(e).__name__, e)
            raise

    async def _arun(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        rate_limiter = get_effective_rate_limiter(config.llm_max_rate, builder)

        if rate_limiter is not None:
            # Use rate limiter to control the rate of requests to the LLM API
            async def summarize_with_rate_limit(results):
                async with rate_limiter:
                    return await summarize_cve(results)

            effective_rate = int(rate_limiter.max_rate / rate_limiter.time_period)
            logger.info("Summarizing %d CVEs with LLM rate limit of %d requests/second",
                        len(state.checklist_results),
                        effective_rate)
            results = await asyncio.gather(*(summarize_with_rate_limit(results)
                                             for results in state.checklist_results.items()))
        else:
            logger.info("Summarizing %d CVEs with no rate limiting", len(state.checklist_results))
            results = await asyncio.gather(*(summarize_cve(results) for results in state.checklist_results.items()))

        state.final_summaries = dict(zip(state.checklist_results.keys(), results))
        return state

    yield FunctionInfo.from_fn(
        _arun,
        input_schema=AgentMorpheusEngineState,
        description=("Generates concise, human-readable summarization paragraph from agent results."))
