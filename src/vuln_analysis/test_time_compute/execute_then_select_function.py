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

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from vuln_analysis.data_models.output import AgentMorpheusOutput

logger = logging.getLogger(__name__)


class ExecuteThenSelectFunctionConfig(FunctionBaseConfig, name="execute_then_select_function"):
    selector: TTCStrategyRef = Field(description="Strategy to select the best output of the function")
    augmented_fn: FunctionRef = Field(description="Function that will be executed")
    output_fn: FunctionRef | None = Field(default=None,
                                          description="Function to output workflow results "
                                          "(e.g. cve_file_output, cve_http_output). "
                                          " If None, only prints to console")
    num_executions: int = Field(3, ge=1, description="Number of times to execute the function")
    max_concurrency: int | None = Field(ge=1,
                                        default=None,
                                        description="The maximum number of concurrent invocations of augmented_fn. "
                                        "None means no rate limiting.")
    early_stop_threshold: int | bool = Field(
        False,
        description=
        """The number of consecutive same outputs to trigger early stopping. The early stop threshold must be strictly
        less than the number of executions. To disable early stopping, set early_stop_threshold to false.""")

    @field_validator("early_stop_threshold")
    @classmethod
    def validate_early_stop_threshold(cls, v):
        """Validate that early_stop_threshold is either a positive integer or False."""
        if (isinstance(v, bool) and v is True) or (not isinstance(v, bool) and v < 1):
            raise ValueError("early_stop_threshold must be a positive integer or False.")
        return v

    @model_validator(mode="after")
    def validate_early_stop_threshold_lt_num_executions(self):
        """Validate that early_stop_threshold is strictly less than num_executions."""
        if not isinstance(self.early_stop_threshold, bool):
            if self.early_stop_threshold >= self.num_executions:
                raise ValueError(f"early_stop_threshold ({self.early_stop_threshold}) must be strictly less than "
                                 f"num_executions ({self.num_executions})")
        return self


@register_function(config_type=ExecuteThenSelectFunctionConfig)
async def execute_then_select_function(config: ExecuteThenSelectFunctionConfig, builder: Builder):
    import asyncio
    import warnings

    executable_fn: Function = await builder.get_function(name=config.augmented_fn)
    output_fn: Function | None = await builder.get_function(name=config.output_fn) if config.output_fn else None

    selector = await builder.get_ttc_strategy(strategy_name=config.selector,
                                              pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
                                              stage_type=StageTypeEnum.SELECTION)

    if executable_fn.has_streaming_output:
        warnings.warn("Streaming output is not supported for this function. "
                      "The function will be executed in non-streaming mode.")

    def _get_justification_statuses(ttc_items: list[TTCItem], vuln_id: str) -> list[str]:
        """Get the justification statuses for a given vuln_id from the TTC items."""

        # Build list of justification statuses from all TTC executions for the given vuln_id
        justification_statuses = []
        for item in ttc_items:
            assert isinstance(item.output, AgentMorpheusOutput), \
                "Item output must be of type AgentMorpheusOutput"

            # Look up the justification status for the given vuln_id
            for output in item.output.output:
                if output.vuln_id == vuln_id:
                    justification_statuses.append(output.justification.status)
                    break

        return justification_statuses

    def _is_early_stopped(affected_statuses: list[str], early_stop_threshold) -> bool:
        """Check if early stopping should occur based on the affected statuses."""

        # Get selection mode from the selector config
        selection_mode = getattr(selector.config, "selection_mode", None)

        # For decisive mode, drop UNKNOWNs before checking for consecutive same outputs
        if selection_mode == "decisive":
            statuses = [status for status in affected_statuses if status != "UNKNOWN"]
        else:
            statuses = affected_statuses

        # Check if we have early_stop_threshold+ consecutive same outputs
        return len(statuses) >= early_stop_threshold and len(set(statuses[-early_stop_threshold:])) == 1

    async def _wrapped_ainvoke(input_msg: executable_fn.input_type, semaphore) -> executable_fn.single_output_type:
        """Wrapper around the executable function's ainvoke to handle concurrency limits."""
        if semaphore:
            async with semaphore:
                return await executable_fn.ainvoke(input_msg)
        else:
            return await executable_fn.ainvoke(input_msg)

    async def _execute_parallel(input_msg: executable_fn.input_type, num_executions: int) -> list[TTCItem]:
        """Execute a specified number of runs in parallel.

        Args:
            input_msg: The input message to pass to the function
            num_executions: Number of parallel executions to perform

        Returns:
            List of TTCItems containing execution results
        """
        semaphore = asyncio.Semaphore(config.max_concurrency) if config.max_concurrency else None

        tasks = [_wrapped_ainvoke(input_msg, semaphore) for _ in range(num_executions)]
        results = await asyncio.gather(*tasks)
        return [TTCItem(input=input_msg, output=res) for res in results]

    async def _execute_with_early_stopping(input_msg: executable_fn.input_type,
                                           early_stop_threshold: int,
                                           num_executions: int) -> list[TTCItem]:
        """Execute function with early stopping after 3 consecutive same outputs.

        Args:
            input_msg: The input message to pass to the function
            num_executions: Number of total executions to perform

        Returns:
            List of TTCItems containing execution results
        """

        # Run the first early_stop_threshold executions in parallel
        ttc_items = await _execute_parallel(input_msg, early_stop_threshold)

        # Run the remaining executions sequentially for vuln_ids that have not been stopped
        current_input = input_msg.model_copy(deep=True)
        for _ in range(early_stop_threshold, num_executions):

            # Check if early stopping should occur for each vuln_id
            vuln_ids = [vuln.vuln_id for vuln in current_input.scan.vulns]
            for vuln_id in vuln_ids:
                justification_statuses = _get_justification_statuses(ttc_items, vuln_id)

                if _is_early_stopped(justification_statuses, early_stop_threshold):
                    # Update metadata for the vuln_id
                    ttc_items[-1].metadata = ttc_items[-1].metadata or {}
                    ttc_items[-1].metadata[vuln_id] = {
                        "early_stopping": True,
                        "early_stopping_execution_num": len(ttc_items),
                        "early_stopping_value": justification_statuses[-1]
                    }

                    # Remove the vuln_id from the current input
                    updated_vulns = [
                        vuln_info for vuln_info in current_input.scan.vulns if vuln_info.vuln_id != vuln_id
                    ]
                    current_input.scan.vulns = updated_vulns
                    logger.info("Stopping early for vuln_id %s after %d executions with affected status %s",
                                vuln_id,
                                len(ttc_items),
                                justification_statuses[-1])

            # Break if there are no more vuln_ids to execute
            if not current_input.scan.vulns:
                logger.info("All vuln_ids have been early stopped after %d executions. Exiting.", len(ttc_items))
                break

            result = await executable_fn.ainvoke(current_input)
            ttc_items.append(TTCItem(input=current_input.model_copy(deep=True), output=result))

        return ttc_items

    async def execute_fn(input_msg: executable_fn.input_type) -> executable_fn.single_output_type:

        if config.max_concurrency:
            logger.info("Executing function %d times with max concurrency of %d",
                        config.num_executions,
                        config.max_concurrency)
        else:
            logger.info("Executing function %d times with no concurrency limit", config.num_executions)

        if config.early_stop_threshold:
            logger.info("Early stopping is enabled with early_stop_threshold=%d", config.early_stop_threshold)
            ttc_items = await _execute_with_early_stopping(input_msg,
                                                           config.early_stop_threshold,
                                                           config.num_executions)
        else:
            ttc_items = await _execute_parallel(input_msg, config.num_executions)

        logger.info("Beginning selection using %s", config.selector)
        selected_items = await selector.ainvoke(items=ttc_items, original_prompt=input_msg)

        # Validate the selected items list
        if not isinstance(selected_items, list):
            raise ValueError("Selected items must be a list.")
        if len(selected_items) < 1:
            raise ValueError("No items were selected. Please check your selector strategy.")
        if not isinstance(selected_items[0], TTCItem):
            raise ValueError("Selected items must be a list of TTCItem objects.")

        if len(selected_items) > 1:
            logger.warning("Multiple items were selected. Returning the first item.")

        if output_fn is not None:
            # Pass the selected output through the output function
            await output_fn.ainvoke(selected_items[0].output)

        return selected_items[0].output

    yield FunctionInfo.from_fn(
        fn=execute_fn,
        description=("This function executes a given function multiple times"
                     "and selects the best output based on the specified selection strategy."),
        converters=executable_fn._converter_list,
    )
