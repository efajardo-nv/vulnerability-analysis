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

import logging
from datetime import datetime
from io import TextIOWrapper
from typing import Literal

# Apply NAT logging patch early to truncate long error messages
from vuln_analysis.utils.nat_logging_patch import apply_nat_logging_patch

apply_nat_logging_patch()

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from vuln_analysis.data_models.input import AgentMorpheusEngineInput
from vuln_analysis.data_models.input import AgentMorpheusInput
from vuln_analysis.data_models.output import AgentMorpheusOutput
from vuln_analysis.data_models.state import AgentMorpheusEngineState
# pylint: disable=unused-import
from vuln_analysis.eval.evaluators import accuracy
from vuln_analysis.eval.evaluators import consistency
from vuln_analysis.functions import cve_agent
from vuln_analysis.functions import cve_check_vuln_deps
from vuln_analysis.functions import cve_checklist
from vuln_analysis.functions import cve_fetch_intel
from vuln_analysis.functions import cve_file_output
from vuln_analysis.functions import cve_generate_vdbs
from vuln_analysis.functions import cve_http_output
from vuln_analysis.functions import cve_justify
from vuln_analysis.functions import cve_process_sbom
from vuln_analysis.functions import cve_summarize
from vuln_analysis.test_time_compute import execute_then_select_function
from vuln_analysis.test_time_compute import majority_voting_selector
from vuln_analysis.tools import lexical_full_search
from vuln_analysis.tools import local_vdb
from vuln_analysis.tools import serp
# pylint: enable=unused-import
from vuln_analysis.utils.concurrency import ctx_parent_max_rate
from vuln_analysis.utils.llm_engine_utils import postprocess_engine_output
from vuln_analysis.utils.llm_engine_utils import preprocess_engine_input

logger = logging.getLogger(__name__)


class CVEAgentWorkflowConfig(FunctionBaseConfig, name="cve_agent"):
    """
    Defines the workflow function for determining the impact of a documented CVEs on a specific project or container.
    """
    cve_generate_vdbs_name: FunctionRef = Field(description="Function name to generate vector databases")
    cve_fetch_intel_name: FunctionRef = Field(description="Function name to fetch intel")
    cve_process_sbom_name: FunctionRef = Field(description="Function name to process SBOMs")
    cve_check_vuln_deps_name: FunctionRef = Field(description="Function name to check vulnerable dependencies")
    cve_checklist_name: FunctionRef = Field(description="Function name to generate checklist")
    cve_agent_executor_name: FunctionRef = Field(description="Function name to run CVE agent on checklist")
    cve_summarize_name: FunctionRef = Field(description="Function name to generate summary")
    cve_justify_name: FunctionRef = Field(description="Function to generate justifications for each CVE")
    cve_output_config_name: FunctionRef | None = Field(default=None,
                                                       description="Function to output workflow results "
                                                       "(e.g. cve_file_output, cve_http_output). "
                                                       " If None, only prints to console")
    description: str = Field(default="Vulnerability analysis for container security workflow",
                             description="Workflow function description")
    missing_source_action: Literal["error", "skip_agent", "continue_with_warning"] = Field(
        default="continue_with_warning",
        description="Action when source analysis is unavailable in the agent due to: "
        "missing source_info, VDB generation failures, or inaccessible repositories\n"
        "     - error: Fail pipeline with validation error\n"
        "     - skip_agent: Collect intel and check dependencies, but skip agent\n"
        "     - continue_with_warning: Run full pipeline with warning log (degraded analysis quality)")
    llm_max_rate: int | None = Field(
        default=None,
        description="Controls the maximum LLM rate limit (requests per second) globally for all workflow functions."
        "Individual function llm_max_rate settings override this value. None means no rate limiting.")


@register_function(config_type=CVEAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def cve_agent_workflow(config: CVEAgentWorkflowConfig, builder: Builder):

    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph

    # Access functions that will be used in this workflow
    cve_generate_vdbs_fn = await builder.get_function(name=config.cve_generate_vdbs_name)
    cve_fetch_intel_fn = await builder.get_function(name=config.cve_fetch_intel_name)
    cve_process_sbom_fn = await builder.get_function(name=config.cve_process_sbom_name)
    cve_check_vuln_deps_fn = await builder.get_function(name=config.cve_check_vuln_deps_name)
    cve_checklist_fn = await builder.get_function(name=config.cve_checklist_name)
    cve_agent_executor_fn = await builder.get_function(name=config.cve_agent_executor_name)
    cve_summary_fn = await builder.get_function(name=config.cve_summarize_name)
    cve_justify_fn = await builder.get_function(name=config.cve_justify_name)
    cve_output_fn = await builder.get_function(
        name=config.cve_output_config_name) if config.cve_output_config_name else None

    # Define langgraph node functions
    async def validate_input_node(state: AgentMorpheusInput) -> AgentMorpheusInput:
        """Validate input based on workflow configuration"""
        if not state.image.source_info and config.missing_source_action == 'error':
            raise ValueError("source_info is required but missing or empty. "
                             "Please provide source code repository information or change "
                             "missing_source_action configuration to 'skip_agent' or 'continue_with_warning'.")
        return state

    async def add_start_time_node(state: AgentMorpheusInput) -> AgentMorpheusInput:
        """Adds the start time to the input"""
        state.scan.started_at = datetime.now().isoformat()
        return state

    async def generate_vdbs_node(state: AgentMorpheusInput) -> AgentMorpheusEngineInput:
        """Generates VDBs based on CVE input"""

        return await cve_generate_vdbs_fn.ainvoke(state.model_dump())

    async def fetch_intel_node(state: AgentMorpheusEngineInput) -> AgentMorpheusEngineInput:
        """Fetch intel for CVE input"""

        return await cve_fetch_intel_fn.ainvoke(state.model_dump())

    async def process_sbom_node(state: AgentMorpheusEngineInput) -> AgentMorpheusEngineInput:
        """Process SBOMs for CVE input"""

        return await cve_process_sbom_fn.ainvoke(state.model_dump())

    async def check_vuln_deps_node(state: AgentMorpheusEngineInput) -> AgentMorpheusEngineInput:
        """Check for vulnerable dependencies"""

        return await cve_check_vuln_deps_fn.ainvoke(state.model_dump())

    async def checklist_node(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        """Generates a checklist based on CVE input"""

        return await cve_checklist_fn.ainvoke(state.model_dump())

    async def agent_executor_node(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        """Executes the checklist using an agent with  ReAct prompt."""

        return await cve_agent_executor_fn.ainvoke(state.model_dump())

    async def summarize_node(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        """Summarizes the results of the execution"""

        return await cve_summary_fn.ainvoke(state.model_dump())

    async def justify_node(state: AgentMorpheusEngineState) -> AgentMorpheusEngineState:
        """Generates a justification for the final summary"""

        return await cve_justify_fn.ainvoke(state.model_dump())

    async def add_completed_time_node(state: AgentMorpheusOutput) -> AgentMorpheusOutput:
        """Adds the completed time to the output"""
        state.input.scan.completed_at = datetime.now().isoformat()
        return state

    async def output_results_node(state: AgentMorpheusOutput) -> AgentMorpheusOutput:
        """Outputs results using configured output function"""

        return await cve_output_fn.ainvoke(state.model_dump()) if cve_output_fn else state

    # define langgraph

    # build llm engine subgraph
    subgraph_builder = StateGraph(AgentMorpheusEngineState)
    subgraph_builder.add_node("checklist", checklist_node)
    subgraph_builder.add_node("agent_executor", agent_executor_node)
    subgraph_builder.add_node("summarize", summarize_node)
    subgraph_builder.add_node("justify", justify_node)

    subgraph_builder.add_edge(START, "checklist")
    subgraph_builder.add_edge("checklist", "agent_executor")
    subgraph_builder.add_edge("agent_executor", "summarize")
    subgraph_builder.add_edge("summarize", "justify")
    subgraph = subgraph_builder.compile()

    async def call_llm_engine_subgraph_node(state: AgentMorpheusEngineInput):

        subgraph_input = preprocess_engine_input(state, config.missing_source_action)
        results = await subgraph.ainvoke(subgraph_input)
        subgraph_output = AgentMorpheusEngineState(**results)
        output = postprocess_engine_output(state, subgraph_output)
        return output

    # build parent graph
    graph_builder = StateGraph(AgentMorpheusOutput, input_schema=AgentMorpheusInput)
    graph_builder.add_node("validate_input", validate_input_node)
    graph_builder.add_node("add_start_time", add_start_time_node)
    graph_builder.add_node("generate_vdbs", generate_vdbs_node)
    graph_builder.add_node("fetch_intel", fetch_intel_node)
    graph_builder.add_node("process_sbom", process_sbom_node)
    graph_builder.add_node("check_vuln_deps", check_vuln_deps_node)
    graph_builder.add_node("llm_engine", call_llm_engine_subgraph_node)
    graph_builder.add_node("add_completed_time", add_completed_time_node)
    graph_builder.add_node("output_results", output_results_node)

    graph_builder.add_edge(START, "validate_input")
    graph_builder.add_edge("validate_input", "add_start_time")
    graph_builder.add_edge("add_start_time", "generate_vdbs")
    graph_builder.add_edge("generate_vdbs", "fetch_intel")
    graph_builder.add_edge("fetch_intel", "process_sbom")
    graph_builder.add_edge("process_sbom", "check_vuln_deps")
    graph_builder.add_edge("check_vuln_deps", "llm_engine")
    graph_builder.add_edge("llm_engine", "add_completed_time")
    graph_builder.add_edge("add_completed_time", "output_results")
    graph_builder.add_edge("output_results", END)
    graph = graph_builder.compile()

    def convert_str_to_agent_morpheus_input(input_str: str) -> AgentMorpheusInput:
        logger.debug("Converting input to AgentMorpheusInput: %s", input_str)
        try:
            return AgentMorpheusInput.model_validate_json(input_str)
        except Exception as e:
            logger.error("Failed to convert input to AgentMorpheusInput: %s. Your input needs to be a json string.", e)
            raise e

    def convert_textio_to_agent_morpheus_input(input_file: TextIOWrapper) -> AgentMorpheusInput:
        logger.debug("Converting input to AgentMorpheusInput: %s", input_file)
        try:
            data = input_file.read()
            return AgentMorpheusInput.model_validate_json(data)
        except Exception as e:
            logger.error(
                "Failed to convert input to AgentMorpheusInput: %s. Your input needs to be a TextIOWrapper object.", e)
            raise e

    def convert_agent_morpheus_output_to_str(output: AgentMorpheusOutput) -> str:
        logger.debug("Converting AgentMorpheusOutput to str: %s", output)
        try:
            return output.model_dump_json()
        except Exception as e:
            logger.error("Failed to convert output to str: %s. Your input needs to be an AgentMorpheusOutput object.",
                         e)
            raise e

    async def _response_fn(input_message: AgentMorpheusInput) -> AgentMorpheusOutput:
        # Set parent workflow's llm_max_rate in context so child functions can access it
        # This is important when cve_agent is used as a function inside another workflow (e.g., TTC)
        token = ctx_parent_max_rate.set(config.llm_max_rate)
        try:
            results = await graph.ainvoke(input_message)
            graph_output = AgentMorpheusOutput(**results)
            return graph_output
        finally:
            # Reset context variable
            ctx_parent_max_rate.reset(token)

    try:
        yield FunctionInfo.from_fn(_response_fn,
                                   description=config.description,
                                   input_schema=AgentMorpheusInput,
                                   converters=[
                                       convert_str_to_agent_morpheus_input,
                                       convert_textio_to_agent_morpheus_input,
                                       convert_agent_morpheus_output_to_str
                                   ])
    except GeneratorExit:
        logger.info("Workflow exited early!")
    finally:
        logger.info("Cleaning up cve-agent workflow.")
