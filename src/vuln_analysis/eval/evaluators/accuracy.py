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
from typing import Literal

import pandas as pd
from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry
from pydantic import Field
from tqdm import tqdm

_STATUS_MAP = {"TRUE": "AFFECTED", "FALSE": "NOT AFFECTED", "UNKNOWN": "UNKNOWN"}


class AccuracyEvaluatorConfig(EvaluatorBaseConfig, name="accuracy"):
    """Configuration for unified accuracy evaluator"""
    field: typing.Literal["status", "label"] = Field(default="status", description="The field to evaluate accuracy on.")
    duplicates_policy: typing.Literal["drop_all", "keep_first", "keep_all"] = Field(
        default="keep_first", description="Policy for handling duplicate CVEs in ground truth.")


@register_evaluator(config_type=AccuracyEvaluatorConfig)
async def accuracy_evaluator(config: AccuracyEvaluatorConfig, builder: EvalBuilder):
    """Register unified accuracy evaluator"""
    evaluator = AccuracyEvaluator(builder.get_max_concurrency(),
                                  field=config.field,
                                  duplicates_policy=config.duplicates_policy)
    yield EvaluatorInfo(config=config,
                        evaluate_fn=evaluator.evaluate,
                        description=f"{config.field.title()} Accuracy Evaluator")


class AccuracyEvaluator:
    '''Configurable accuracy evaluator for either the status or label field.'''

    def __init__(self,
                 max_concurrency: int,
                 field: str = "status",
                 duplicates_policy: Literal["drop_all", "keep_first", "keep_all"] = "keep_first"):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        # field must be either "status" or "label"
        self.field = field
        # duplicates_policy: "drop_all" | "keep_first" | "keep_all"
        self.duplicates_policy = duplicates_policy

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:

        def _preprocess_ground_truth(ground_truth_list, field: str, policy: str):
            """Build a mapping of vuln_id -> acceptable answer set based on policy.

            - drop_all: only vuln_ids with a single unique answer are kept
            - keep_first: take the first encountered answer
            - keep_all: any answer in ground truth is considered acceptable
            """
            # Collect ground truth answers per vuln_id
            answers_per_id = {}
            for gt in ground_truth_list:
                vuln_id = gt.get("vuln_id")
                if not vuln_id:
                    continue
                ans = gt.get(field)
                if ans is None:
                    continue
                answers_per_id.setdefault(vuln_id, [])
                answers_per_id[vuln_id].append(ans)

            acceptable_map = {}

            if policy == "drop_all":
                for vuln_id, answers in answers_per_id.items():
                    distinct = list(dict.fromkeys(answers))  # preserve order, unique
                    if len(distinct) == 1:
                        acceptable_map[vuln_id] = {distinct[0]}
                return acceptable_map

            if policy == "keep_first":
                # Keep first encountered answer per vuln_id
                for vuln_id, answers in answers_per_id.items():
                    first_answer = answers[0]
                    acceptable_map[vuln_id] = {first_answer}
                return acceptable_map

            # keep_all: any of the distinct answers are acceptable
            for vuln_id, answers in answers_per_id.items():
                acceptable_map[vuln_id] = set(answers)
            return acceptable_map

        def _extract_rep_number(item_id: str) -> str:
            """Extract rep number from item id (e.g., 'container1_rep1' -> 'rep1')"""
            if "_rep" in item_id:
                return "rep" + item_id.split("_rep")[-1]
            return "rep1"  # Default to rep1 if no rep suffix

        def _group_by_rep(items: list[EvalInputItem]) -> dict[str, list[EvalInputItem]]:
            """Group items by rep number"""
            groups = {}
            for item in items:
                rep_id = _extract_rep_number(item.id)
                groups.setdefault(rep_id, [])
                groups[rep_id].append(item)
            return groups

        async def process_rep_group(items: list[EvalInputItem]):
            """Process all containers in a single rep together"""
            # Aggregate results across all containers in this rep
            all_per_item_accuracies = []
            acceptable_answers_map_combined = {}
            test_set_file = "unknown"
            container_data = {}

            if self.field == "status":
                answer_key = "status_answer"
                gen_key = "generated_status_answer"
            else:
                answer_key = "label_answer"
                gen_key = "generated_label_answer"

            for item in items:
                # Extract container name from item id (e.g., "container1_rep1" -> "container1")
                container_name = item.id.split('_rep')[0] if '_rep' in item.id else item.id

                # Build acceptable answers map from the original ground truth list
                original_ground_truth = item.full_dataset_entry.get("ground_truth", [])
                acceptable_answers_map = _preprocess_ground_truth(original_ground_truth, self.field, self.duplicates_policy)
                acceptable_answers_map_combined.update(acceptable_answers_map)

                pipeline_result = json.loads(item.output_obj)["output"]
                test_set_file = item.full_dataset_entry.get("test_set_file", "unknown")

                # Track data per container
                container_questions = []
                container_generated_answers = []

                for result in pipeline_result:
                    vuln_id = result["vuln_id"]

                    if self.field == "status":
                        justification_status = result.get("justification", {}).get("status", "MISSING")
                        pred = _STATUS_MAP.get(justification_status, "MISSING")
                        acceptable_set = acceptable_answers_map.get(vuln_id)
                    else:  # label
                        justification_label = result.get("justification", {}).get("label", "MISSING")
                        pred = justification_label
                        acceptable_set = acceptable_answers_map.get(vuln_id)

                    # If this vuln_id was filtered out by the duplicates policy, skip
                    if acceptable_set is None:
                        continue

                    container_questions.append(vuln_id)
                    container_generated_answers.append(pred)
                    all_per_item_accuracies.append(float(pred in acceptable_set))

                # Output the data per-container
                container_data[container_name] = {
                    "question": container_questions,
                    answer_key: [sorted(list(acceptable_answers_map_combined.get(q, set()))) for q in container_questions],
                    gen_key: container_generated_answers,
                }

            total_correct = sum(all_per_item_accuracies)
            total_items = len(all_per_item_accuracies)
            avg = round(total_correct / total_items, 2) if total_items > 0 else 0.0
            reasoning = {**container_data, "test_set_file": test_set_file}

            return avg, reasoning, total_items, total_correct

        async def wrapped_process_rep(rep_id: str, items: list[EvalInputItem]) -> tuple[str, float, dict, int, int]:
            async with self.semaphore:
                result = await process_rep_group(items)
                pbar.update(len(items))
                return (rep_id, *result)

        # Group items by rep number
        rep_groups = _group_by_rep(eval_input.eval_input_items)

        try:
            tqdm_position = TqdmPositionRegistry.claim()
            pbar = tqdm(total=len(eval_input.eval_input_items),
                        desc=f"Evaluating {self.field.title()} Accuracy",
                        position=tqdm_position)
            results = await asyncio.gather(*[wrapped_process_rep(rep_id, items) for rep_id, items in rep_groups.items()])
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        # Unpack results: rep_id, avg, reasoning, total_items, total_correct
        if results:
            rep_ids, all_averages, all_reasonings, _, _ = zip(*results)
        else:
            rep_ids, all_averages, all_reasonings = [], [], []

        # Output dict format with .describe() summary
        scores_sequential = list(all_averages)
        if scores_sequential:
            series = pd.Series(scores_sequential)
            desc = series.describe()  # count, mean, std, min, 25%, 50%, 75%, max
            # Round values
            final_output = {k: (int(v) if k == "count" else round(float(v), 2)) for k, v in desc.items()}
        else:
            final_output = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

        eval_output_items = [
            EvalOutputItem(id=rep_id, score=score, reasoning=reasoning)
            for rep_id, score, reasoning in zip(rep_ids, all_averages, all_reasonings)
        ]

        return EvalOutput(average_score=final_output, eval_output_items=eval_output_items)
