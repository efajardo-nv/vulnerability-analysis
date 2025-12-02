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

import typing

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from pydantic import Field

from vuln_analysis.data_models.output import AgentMorpheusEngineOutput
from vuln_analysis.data_models.output import AgentMorpheusOutput


class MajorityVotingSelectionConfig(TTCStrategyBaseConfig, name="majority_voting_selection"):
    """
    Configuration for Majority Voting Selection
    """
    selection_mode: typing.Literal["simple", "decisive"] = Field(description="The majority voting selection mode.",
                                                                 default="decisive")


class MajorityVotingSelector(StrategyBase):

    async def build_components(self, builder: Builder) -> None:
        pass

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.AGENT_EXECUTION]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        if len(items) == 0:
            return []

        results = [item.output for item in items]

        # Get value counts of the justification status for each vuln_id
        assert isinstance(results[0], AgentMorpheusOutput), "Results must be of type AgentMorpheusOutput"
        value_counts = {output.vuln_id: {} for output in results[0].output}
        for result in results:
            assert isinstance(result, AgentMorpheusOutput), "Result must be of type AgentMorpheusOutput"
            for output in result.output:
                vuln_id = output.vuln_id
                value = output.justification.status
                value_counts[vuln_id][value] = value_counts.get(vuln_id, {}).get(value, 0) + 1

        # Find the most common value for each vuln_id
        majority_values = {}
        for vuln_id, counts in value_counts.items():

            assert isinstance(self.config, MajorityVotingSelectionConfig), \
                "Config must be of type MajorityVotingSelectionConfig"

            if self.config.selection_mode == "simple":
                max_value = max(counts.values())
                candidates = [candidate for candidate, votes in counts.items() if votes == max_value]
                # Break ties as 'UNKNOWN'
                if len(candidates) == 1:
                    majority_values[vuln_id] = candidates[0]
                else:
                    majority_values[vuln_id] = 'UNKNOWN'

            elif self.config.selection_mode == "decisive":
                # Prefer 'TRUE' or 'FALSE' if there is a clear majority
                if counts.get("TRUE", 0) > counts.get("FALSE", 0):
                    majority_values[vuln_id] = "TRUE"
                elif counts.get("TRUE", 0) < counts.get("FALSE", 0):
                    majority_values[vuln_id] = "FALSE"
                # Break ties as 'UNKNOWN' if there was a run with 'UNKNOWN' status
                # Otherwise, default to 'TRUE' to be conservative
                else:
                    if "UNKNOWN" in counts:
                        majority_values[vuln_id] = "UNKNOWN"
                    else:
                        majority_values[vuln_id] = "TRUE"
            else:
                raise ValueError(f"Invalid selection mode: {self.config.selection_mode}")

        # Check for early stopping metadata
        early_stopped_values = {}
        for item in items:
            if item.metadata:
                for vuln_id, meta in item.metadata.items():
                    if meta.get('early_stopping', False):
                        early_stopped_values[vuln_id] = meta['early_stopping_value']

        # Combine majority values with early stopped values, giving precedence to early stopped values
        final_values = {**majority_values, **early_stopped_values}

        # Select the first result matching the final value for each vuln_id
        new_ttc_item = items[0]
        new_output = []

        for vuln_id, final_value in final_values.items():
            for item in items:
                assert isinstance(item.output, AgentMorpheusOutput), "Item output must be of type AgentMorpheusOutput"

                # Create dict for easy output retrieval by vuln_id
                output_dict = {output.vuln_id: output for output in item.output.output}

                if output_dict[vuln_id].justification.status == final_value:
                    new_engine_output = AgentMorpheusEngineOutput(vuln_id=vuln_id,
                                                                  checklist=output_dict[vuln_id].checklist,
                                                                  summary=output_dict[vuln_id].summary,
                                                                  justification=output_dict[vuln_id].justification)
                    new_output.append(new_engine_output)
                    break

        assert isinstance(new_ttc_item.output, AgentMorpheusOutput), "TTC output must be of type AgentMorpheusOutput"
        new_ttc_item.output.output = new_output

        return [new_ttc_item]


@register_ttc_strategy(config_type=MajorityVotingSelectionConfig)
async def register_majority_voting_selector(config: MajorityVotingSelectionConfig, builder: Builder):
    """
    Register the MajorityVotingSelector.
    """
    selector = MajorityVotingSelector(config)
    yield selector
