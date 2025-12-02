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
import json
import typing
from collections import Counter

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry
from pydantic import Field
from tqdm import tqdm


class ConsistencyEvaluatorConfig(EvaluatorBaseConfig, name="consistency"):
    """Configuration for consistency evaluator"""
    field: typing.Literal["label", "status"] = Field(default="label",
                                                     description="The field to evaluate consistency on.")


@register_evaluator(config_type=ConsistencyEvaluatorConfig)
async def consistency_evaluator(config: ConsistencyEvaluatorConfig, builder: EvalBuilder):
    """Register consistency evaluator"""
    evaluator = ConsistencyEvaluator(builder.get_max_concurrency(), field=config.field)
    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluator.evaluate,
                        description=f"{config.field.title()} Consistency Evaluator")


class ConsistencyEvaluator:
    '''Evaluator class for measuring consistency of the specified field via the agreement probability metric ∑p², where p is the probability of each unique value'''

    def __init__(self, max_concurrency: int, field: str = "label"):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.field = field  # "label" or "status"

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        '''Evaluate consistency across multiple runs per CVE, broken down by container'''

        async def extract_run_data(item):
            """Extract data from a single evaluation run"""
            async with self.semaphore:
                evaluation_run = json.loads(item.output_obj)['output']
                pbar.update(1)
                return evaluation_run, item

        try:
            tqdm_position = TqdmPositionRegistry.claim()
            pbar = tqdm(total=len(eval_input.eval_input_items), desc="Evaluating Consistency", position=tqdm_position)

            # Extract data from all evaluation runs in parallel
            all_runs_with_items = await asyncio.gather(
                *[extract_run_data(item) for item in eval_input.eval_input_items])
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        # Group runs by container (extract container name from item id)
        container_runs = {}
        for evaluation_run, item in all_runs_with_items:
            container_name = item.id.split('_rep')[0]

            if container_name not in container_runs:
                container_runs[container_name] = []
            container_runs[container_name].append(evaluation_run)

        # Validate that each container has multiple reps
        for container_name, runs in container_runs.items():
            if len(runs) <= 1:
                raise ValueError(
                    f"Consistency evaluator requires multiple runs per container. Container '{container_name}' has only {len(runs)} run(s). Consistency cannot be measured with a single run per container."
                )

        # Calculate consistency for each container
        eval_output_items = []
        all_cve_consistencies = []  # Collect all CVE consistency scores across all containers

        for container_name, runs in container_runs.items():
            # Aggregate results across all reps for this container
            vuln_responses = {}
            for evaluation_run in runs:
                for cve_result in evaluation_run:  # Group by CVE
                    vuln_id = cve_result['vuln_id']
                    # Measure consistency on the specified field
                    field_value = cve_result.get('justification', {}).get(self.field, 'MISSING')

                    if vuln_id not in vuln_responses:
                        vuln_responses[vuln_id] = []

                    vuln_responses[vuln_id].append(field_value)

            # Calculate consistency score per CVE for this container
            consistencies_dict = {}  # Stores consistency scores mapped to CVE
            consistencies = []  # Stores all consistency scores in an array

            for vuln_id, values in vuln_responses.items():
                if len(values) > 1:
                    value_counts = Counter(values)
                    n = len(values)
                    probs = [count / n for count in value_counts.values()]
                    consistency = sum(p * p for p in probs)
                else:
                    consistency = 1.0  # Perfect consistency if there is only one response

                consistencies_dict[vuln_id] = round(consistency, 2)
                consistencies.append(round(consistency, 2))

            # Add all CVE consistency scores from this container to the global collection
            all_cve_consistencies.extend(consistencies)

            # Calculate average consistency for this container
            container_avg = sum(consistencies) / len(consistencies) if consistencies else 1.0
            container_score = round(container_avg, 2)

            # Create output with per-CVE scores for this container
            output = {
                f"{self.field}_consistencies": consistencies_dict,
                "total_cves": len(vuln_responses),
                "num_reps": len(runs),
            }

            # Create EvalOutputItem for this container
            eval_output_items.append(
                EvalOutputItem(id=f"{self.field.title()} Consistency per CVE - {container_name}",
                               score=container_score,
                               reasoning=output))

        # Calculate overall average across all CVEs from all containers
        overall_avg = sum(all_cve_consistencies) / len(all_cve_consistencies) if all_cve_consistencies else 1.0
        overall_score = round(overall_avg, 2)

        return EvalOutput(average_score=overall_score, eval_output_items=eval_output_items)
