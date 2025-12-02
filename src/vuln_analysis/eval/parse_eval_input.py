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

import json
from pathlib import Path

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem

from vuln_analysis.data_models.eval_input import AgentMorpheusEvalDataset


def parse_input(file_path: Path) -> EvalInput:
    """
    Transform a human-readable eval dataset in the eval_datasets/ directory into a series of EvalInputItems
    that the NAT evaluation harness accepts. Each EvalInputItem corresponds to a container.
    EvalInputItem `id` is the container name, `question` encodes the source code and input CVEs,
    and `answer` encodes the CVE ground truth mapping.

    Usage (outside of the full NAT pipeline):
        python3 parse_eval_input.py [test_set_file].json --preview -o [your output file].json
        Example:
        python3 parse_eval_input.py ../data/eval_datasets/eval_dataset.json --preview -o my_eval_string.json
    Run this inside the vuln_analysis container, from the eval directory.

    Example output:
        [
            {
                "id": "morpheus:23.11-runtime",
                "question": "{\"image\":{\"name\":\"nvcr.io/nvidia/morpheus/morpheus\",\"tag\":\"23.11-runtime\",\"source_info\":[{\"type\":\"code\",\"git_repo\":\"https://github.com/nv-morpheus/Morpheus.git\",\"ref\":\"branch-23.11\",\"include\":[\"**/*.cpp\",\"**/*.cu\",\"**/*.cuh\",\"**/*.h\",\"**/*.hpp\",\"**/*.ipynb\",\"**/*.py\",\"**/*Dockerfile\"],\"exclude\":[\"tests/**/*\"]},{\"type\":\"doc\",\"git_repo\":\"https://github.com/nv-morpheus/Morpheus.git\",\"ref\":\"branch-23.11\",\"include\":[\"**/*.md\",\"docs/**/*.rst\"]}],\"sbom_info\":{\"_type\":\"file\",\"file_path\":\"data/sboms/nvcr.io/nvidia/morpheus/morpheus:v23.11.01-runtime.sbom\"}},\"scan\":{\"vulns\":[{\"vuln_id\":\"GHSA-3f63-hfp8-52jq\"},{\"vuln_id\":\"CVE-2023-36632\"}]}}",
                "answer": "{\"GHSA-3f63-hfp8-52jq\":\"NOT AFFECTED\",\"GHSA-3f63-hfp8-52jq_label\":\"code_not_reachable\",\"CVE-2023-36632\":\"AFFECTED\",\"CVE-2023-36632_label\":\"vulnerable\"}"
            },
            {
                "id": "morpheus:24.03-runtime",
                "question": "{\"image\":{\"name\":\"nvcr.io/nvidia/morpheus/morpheus\",\"tag\":\"v24.03.02-runtime\",\"source_info\":[{\"type\":\"code\",\"git_repo\":\"https://github.com/nv-morpheus/Morpheus.git\",\"ref\":\"v24.03.02\",\"include\":[\"**/*.cpp\",\"**/*.cu\",\"**/*.cuh\",\"**/*.h\",\"**/*.hpp\",\"**/*.ipynb\",\"**/*.py\",\"**/*Dockerfile\"],\"exclude\":[\"tests/**/*\"]},{\"type\":\"doc\",\"git_repo\":\"https://github.com/nv-morpheus/Morpheus.git\",\"ref\":\"v24.03.02\",\"include\":[\"**/*.md\",\"docs/**/*.rst\"]}],\"sbom_info\":{\"_type\":\"file\",\"file_path\":\"data/sboms/nvcr.io/nvidia/morpheus/morpheus:v24.03.02-runtime.sbom\"}},\"scan\":{\"vulns\":[{\"vuln_id\":\"GHSA-3f63-hfp8-52jq\"},{\"vuln_id\":\"CVE-2023-36632\"}]}}",
                "answer": "{\"GHSA-3f63-hfp8-52jq\":\"NOT AFFECTED\",\"GHSA-3f63-hfp8-52jq_label\":\"code_not_reachable\",\"CVE-2023-36632\":\"AFFECTED\",\"CVE-2023-36632_label\":\"vulnerable\"}"
            }
        ]
    """
    # Load and validate the test set file using Pydantic models
    with open(file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    dataset = AgentMorpheusEvalDataset.model_validate(config_data)

    # Extract containers
    containers = dataset.containers

    # Get dataset metadata from the file
    dataset_id = dataset.dataset_id
    dataset_description = dataset.dataset_description
    test_set_file = file_path.name  # Extract filename from the file path
    print(f"Processing dataset: {dataset_id} (file: {test_set_file})")

    # Create evaluation items for each container
    eval_items = []

    for container_id, container_data in containers.items():
        container_image = container_data.container_image.model_dump()
        ground_truth = [gt.model_dump() for gt in container_data.ground_truth]

        # Create scan structure from ground truth
        scan_vulns = []
        ground_truth_mapping = {}
        seen_vuln_ids = set()  # Ensures vuln_id uniqueness

        for gt_item in ground_truth:
            vuln_id = gt_item.get('vuln_id')
            status = gt_item.get('status')
            label = gt_item.get('label')

            if not (vuln_id and status):
                continue

            if vuln_id in seen_vuln_ids:  # skip duplicate vuln_ids
                continue

            seen_vuln_ids.add(vuln_id)

            # Add to scan structure
            scan_vulns.append({"vuln_id": vuln_id})

            # Add to ground truth mapping
            ground_truth_mapping[vuln_id] = status

            # Add label to ground truth mapping if available
            if label:
                ground_truth_mapping[f"{vuln_id}_label"] = label

        # Create the question structure (container_image + scan)
        question_structure = {"image": container_image, "scan": {"vulns": scan_vulns}}

        # Convert to stringified format compatible with NAT EvalInputItem format
        question_string = json.dumps(question_structure, separators=(',', ':'))
        answer_string = json.dumps(ground_truth_mapping, separators=(',', ':'))

        # Create evaluation item
        eval_item = EvalInputItem(id=container_id,
                                  input_obj=question_string,
                                  expected_output_obj=answer_string,
                                  full_dataset_entry={
                                      "id": container_id,
                                      "question": question_string,
                                      "answer": answer_string,
                                      "test_set_file": test_set_file,
                                      "dataset_id": dataset_id,
                                      "dataset_description": dataset_description,
                                      "container_id": container_id,
                                      "ground_truth": ground_truth
                                  })

        eval_items.append(eval_item)

    if not eval_items:
        print("No valid evaluation items created")
        return EvalInput(eval_input_items=[])

    print(f"Created {len(eval_items)} evaluation items for dataset '{dataset_id}'")
    return EvalInput(eval_input_items=eval_items)


# For debugging - saves the EvalInputItem string to a file
def save_as_baseline_string_format(eval_input: EvalInput, output_path: Path) -> None:
    baseline_data = []

    for item in eval_input.eval_input_items:
        baseline_item = {"id": item.id, "question": item.input_obj, "answer": item.expected_output_obj}
        baseline_data.append(baseline_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_data, f, indent=4)

    print(f"Saved baseline string format to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse consolidated evaluation dataset with multiple containers and test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_file', help='Path to eval-dataset.json file')
    parser.add_argument('-o', '--output', help='Output path for baseline string format (optional)')
    parser.add_argument('--preview', action='store_true', help='Show preview of first converted item')

    args = parser.parse_args()

    try:
        input_path = Path(args.input_file)

        # Convert to EvalInput format
        eval_input = parse_input(input_path)

        # Save as baseline string format if output is specified
        if args.output:
            output_path = Path(args.output)
            save_as_baseline_string_format(eval_input, output_path)
        else:
            # Default output path in eval directory
            eval_dir = Path(__file__).parent
            default_output = eval_dir / "eval_input_string.json"
            save_as_baseline_string_format(eval_input, default_output)

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
