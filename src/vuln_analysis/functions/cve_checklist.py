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

import pandas as pd
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

logger = logging.getLogger(__name__)


class CVEChecklistToolConfig(FunctionBaseConfig, name="cve_checklist"):
    """
    Defines a function that generates tailored, context-sensitive task checklist for impact analysis.
    """
    llm_name: str = Field(description="The LLM model to use")
    prompt: str | None = Field(
        default=None,
        description=
        "Manually set the prompt for the specific model in the configuration. The prompt can either be passed in as a "
        "string of text or as a path to a text file containing the desired prompting.")
    llm_max_rate: int | None = Field(
        default=None,
        description="Maximum LLM rate limit (requests per second) for checklist generation tasks. "
        "If set to a number, overrides the workflow-level llm_max_rate. If None, inherits from workflow-level setting.")


@register_function(config_type=CVEChecklistToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def cve_checklist(config: CVEChecklistToolConfig, builder: Builder):

    from vuln_analysis.data_models.state import AgentMorpheusEngineState
    from vuln_analysis.utils.checklist_prompt_generator import _parse_list
    from vuln_analysis.utils.checklist_prompt_generator import generate_checklist
    from vuln_analysis.utils.concurrency import get_effective_rate_limiter

    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate_checklist_for_cve(cve_intel):

        checklist = await generate_checklist(prompt=config.prompt,
                                             llm=llm,
                                             input_dict=cve_intel,
                                             enable_llm_list_parsing=False)

        checklist = await _parse_list([checklist])

        return cve_intel["vuln_id"], checklist[0]

    async def _arun(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        intel_df = pd.json_normalize([x.model_dump(mode="json") for x in state.cve_intel], sep="_")
        workflow_cve_intel = intel_df.to_dict(orient='records')

        rate_limiter = get_effective_rate_limiter(config.llm_max_rate, builder)

        if rate_limiter is not None:
            # Use rate limiter to control the rate of requests to the LLM API
            async def generate_with_rate_limit(cve_intel):
                async with rate_limiter:
                    return await generate_checklist_for_cve(cve_intel)

            effective_rate = int(rate_limiter.max_rate / rate_limiter.time_period)
            logger.info("Generating checklists for %d CVEs with LLM rate limit of %d requests/second",
                        len(workflow_cve_intel),
                        effective_rate)
            results = await asyncio.gather(*(generate_with_rate_limit(cve_intel) for cve_intel in workflow_cve_intel))
        else:
            logger.info("Generating checklists for %d CVEs with no rate limiting", len(workflow_cve_intel))
            results = await asyncio.gather(*(generate_checklist_for_cve(cve_intel) for cve_intel in workflow_cve_intel))

        state.checklist_plans = dict(results)
        return state

    yield FunctionInfo.from_fn(
        _arun,
        input_schema=AgentMorpheusEngineState,
        description=("Generates tailored, context-sensitive task checklist for impact analysis."))
