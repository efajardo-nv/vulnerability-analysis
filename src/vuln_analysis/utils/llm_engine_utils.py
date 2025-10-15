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
import typing

from ordered_set import OrderedSet

from vuln_analysis.data_models.input import AgentMorpheusEngineInput
from vuln_analysis.data_models.output import AgentMorpheusEngineOutput
from vuln_analysis.data_models.output import AgentMorpheusOutput
from vuln_analysis.data_models.output import ChecklistItemOutput
from vuln_analysis.data_models.output import JustificationOutput
from vuln_analysis.data_models.state import AgentMorpheusEngineState
from vuln_analysis.data_models.state import LLMEngineSkipReason

logger = logging.getLogger(__name__)


def determine_engine_skip_reasons(message: AgentMorpheusEngineInput,
                                  missing_source_action: str) -> dict[str, LLMEngineSkipReason | None]:
    """
    Determine skip reasons for vulnerabilities in LLM engine processing.

    Args:
        message: The input message containing vulnerability and image information
        missing_source_action: Action to take when source analysis is unavailable

    Returns:
        Dictionary mapping vuln_id to skip reason (None = processable, LLMEngineSkipReason = reason to skip)
    """
    assert message.info.intel is not None, "The input message must have intel information"
    assert len(message.input.scan.vulns) == len(message.info.intel), \
        "The number of intel objects must match the number of vulnerabilities"
    if message.info.vulnerable_dependencies is not None:
        assert len(message.input.scan.vulns) == len(message.info.vulnerable_dependencies), \
            "The number of intel objects must match the number of vulnerabilities"

    skip_reasons = {}
    vuln_ids = [vuln.vuln_id for vuln in message.input.scan.vulns]

    # Check for global skip conditions first

    # Check for SBOM package availability
    if not message.info.sbom or not message.info.sbom.packages:
        return {vuln_id: LLMEngineSkipReason.NO_SBOM_PACKAGES for vuln_id in vuln_ids}

    # Check for source_info and source vdb/index availability
    has_source_info = message.input.image.source_info is not None
    has_vdbs = (message.info.vdb and any(
        [message.info.vdb.code_vdb_path, message.info.vdb.doc_vdb_path, message.info.vdb.code_index_path]))

    if not has_source_info or not has_vdbs:
        if missing_source_action == "error":
            raise RuntimeError(
                "Source info or VDBs are missing and missing_source_action is set to 'error'. "
                "To allow the pipeline to continue, change the setting to 'skip_agent' or 'continue_with_warning'.")
        if missing_source_action == "skip_agent":
            logger.warning("Skipping LLM engine processing for all vulns. Source info or VDBs are missing and "
                           "missing_source_action is set to 'skip_agent'.")
            return {vuln_id: LLMEngineSkipReason.MISSING_SOURCE for vuln_id in vuln_ids}
        if missing_source_action == "continue_with_warning":
            logger.warning("Continuing with LLM engine processing despite missing source info or VDBs since "
                           "missing_source_action is set to 'continue_with_warning'. Analysis will be degraded.")

    # Per-vulnerability checks

    # Check for vulnerable dependencies
    if message.info.vulnerable_dependencies is None:  # VulnerableDependencyChecker was skipped, run all vulnerabilities
        vdc_run_agent = [True for _ in vuln_ids]
    else:
        vdc_run_agent = [
            len(v.vulnerable_sbom_packages) > 0 or len(v.vuln_package_intel_sources) == 0
            for v in message.info.vulnerable_dependencies
        ]
    # Check for sufficient intel
    sufficient_intel_run_agent = [intel.has_sufficient_intel_for_agent for intel in message.info.intel]

    # Determine if each vulnerability should be skipped
    for i, vuln_id in enumerate(vuln_ids):

        # Ensure we only process the first instance of each vuln_id in case of duplicates
        if vuln_id in skip_reasons:
            continue

        has_vulnerable_deps = vdc_run_agent[i]
        has_sufficient_intel = sufficient_intel_run_agent[i]

        if not has_sufficient_intel:
            skip_reasons[vuln_id] = LLMEngineSkipReason.INSUFFICIENT_INTEL
        elif not has_vulnerable_deps:
            skip_reasons[vuln_id] = LLMEngineSkipReason.NO_VULNERABLE_PACKAGES
        else:
            skip_reasons[vuln_id] = None

    return skip_reasons


def preprocess_engine_input(message: AgentMorpheusEngineInput,
                            missing_source_action: str = "continue_with_warning") -> AgentMorpheusEngineState:
    """
    Preprocess the input for the LLM Engine.

    Args:
        message: The input message containing vulnerability and image information
        missing_source_action: Action to take when source analysis is unavailable

    Returns:
        AgentMorpheusEngineState containing inputs that should be processed by the LLM Engine
    """
    assert message.info.intel is not None, "The input message must have intel information"
    skip_reasons = determine_engine_skip_reasons(message, missing_source_action)

    # Get set of vuln_ids that should be processed by the agent (skip_reasons is None)
    vulns_for_agent = {vuln_id for vuln_id, skip_reason in skip_reasons.items() if skip_reason is None}

    # Drop duplicate vuln_ids
    unique_vulns = list(OrderedSet(vulns_for_agent))
    if len(vulns_for_agent) > len(unique_vulns):
        logger.warning(
            "Input contains duplicate vuln_ids. Passing only the first instance of each vuln_id to the LLM Engine.")
        vulns_for_agent = unique_vulns

    # Filter intel for processable vulnerabilities, taking only the first instance of each vuln_id
    filtered_intel = {}
    for i in message.info.intel:
        if i.vuln_id in vulns_for_agent and i.vuln_id not in filtered_intel:
            filtered_intel[i.vuln_id] = i

    logger.info("Passing %d vuln_id(s) to the LLM Engine", len(vulns_for_agent))

    return AgentMorpheusEngineState(code_vdb_path=message.info.vdb.code_vdb_path if message.info.vdb else None,
                                    doc_vdb_path=message.info.vdb.doc_vdb_path if message.info.vdb else None,
                                    code_index_path=message.info.vdb.code_index_path if message.info.vdb else None,
                                    cve_intel=list(filtered_intel.values()),
                                    skip_reasons=skip_reasons)


def parse_agent_morpheus_engine_output(vuln_id: str,
                                       checklist_results: list[dict[str, typing.Any]],
                                       summary: str,
                                       justification: dict[str, str]) -> AgentMorpheusEngineOutput:
    """
    Parse the output fields for a single vulnerability into an AgentMorpheusEngineOutput object.
    """
    # Convert list of checklist item dicts to list of ChecklistItemOutput objects
    checklist_output = [
        ChecklistItemOutput(input=item["input"], response=item["output"], intermediate_steps=item["intermediate_steps"])
        for item in checklist_results
    ]

    # Combine justification model outputs into a single JustificationOutput object
    justification_output = JustificationOutput(label=justification["justification_label"],
                                               reason=justification["justification"],
                                               status=justification["affected_status"])

    return AgentMorpheusEngineOutput(vuln_id=vuln_id,
                                     checklist=checklist_output,
                                     summary=summary,
                                     justification=justification_output)


def build_deficient_intel_output(vuln_id: str) -> AgentMorpheusEngineOutput:
    summary = ("There is insufficient intel available to determine vulnerability. "
               "This is either due to the CVE not existing or there is not enough "
               "gathered intel for the agent to make an informed decision.")
    justification = JustificationOutput(label="insufficient_intel",
                                        reason="Insufficient intel available for CVE",
                                        status="UNKNOWN")
    return AgentMorpheusEngineOutput(
        vuln_id=vuln_id,
        checklist=[
            ChecklistItemOutput(input="Agent bypassed: Insufficient intel gathered. No checklist generated.",
                                response=summary,
                                intermediate_steps=None)
        ],
        summary=summary,
        justification=justification)


def build_no_vuln_packages_output(vuln_id: str) -> AgentMorpheusEngineOutput:
    summary = ("The VulnerableDependencyChecker did not find any vulnerable packages "
               "or dependencies in the SBOM.")
    justification = JustificationOutput(label="false_positive",
                                        reason="No vulnerable packages or dependencies were detected in the SBOM.",
                                        status="FALSE")
    return AgentMorpheusEngineOutput(
        vuln_id=vuln_id,
        checklist=[
            ChecklistItemOutput(input="Agent bypassed: no vulnerable packages detected. Checklist not generated.",
                                response=("The VulnerableDependencyChecker did not find any vulnerable packages "
                                          "or dependencies in the SBOM and so the agent was bypassed."),
                                intermediate_steps=None)
        ],
        summary=summary,
        justification=justification)


def build_no_sbom_output(vuln_id: str) -> AgentMorpheusEngineOutput:
    summary = ("There were no SBOM packages found for the image. This is either due to "
               "an invalid SBOM input or empty SBOM. There is not enough information "
               "to make an informed decision.")
    justification = JustificationOutput(label="no_sbom_packages",
                                        reason="No SBOM packages found for image.",
                                        status="UNKNOWN")
    return AgentMorpheusEngineOutput(vuln_id=vuln_id,
                                     checklist=[
                                         ChecklistItemOutput(
                                             input="Agent bypassed: no SBOM packages found. Checklist not generated.",
                                             response=summary,
                                             intermediate_steps=None)
                                     ],
                                     summary=summary,
                                     justification=justification)


def build_missing_source_action_output(vuln_id: str) -> AgentMorpheusEngineOutput:
    summary = ("Analysis is unavailable due to missing source_info, inaccessible repositories, or missing VDBs. "
               "Intel was collected and dependency checking was performed but agent analysis was skipped. "
               "Please ensure valid source_info for comprehensive code analysis.")
    justification = JustificationOutput(label="missing_source",
                                        reason="Missing source info, inaccessible repositories, or missing VDBs.",
                                        status="UNKNOWN")
    return AgentMorpheusEngineOutput(
        vuln_id=vuln_id,
        checklist=[
            ChecklistItemOutput(
                input="Agent bypassed: analysis unavailable due to missing source_info, inaccessible repositories, "
                "or missing VDBs. Checklist not generated.",
                response=summary,
                intermediate_steps=None)
        ],
        summary=summary,
        justification=justification)


def postprocess_engine_output(message: AgentMorpheusEngineInput,
                              result: AgentMorpheusEngineState) -> AgentMorpheusOutput:
    """
    Postprocess the engine output

    Args:
        message: The input message containing vulnerability and image information
        result: The LLM Engine output results

    Returns:
        Final AgentMorpheusOutput object containing results for all vulnerabilities, including ones that were skipped
    """
    input_vuln_ids = [vuln.vuln_id for vuln in message.input.scan.vulns]

    # For each vuln_id, get LLM Engine output if it exists
    # or create placeholder output if it skipped the workflow
    output: list[AgentMorpheusEngineOutput] = []
    engine_output_vuln_ids = list(result.final_summaries.keys())
    skip_reasons = result.skip_reasons

    for vuln_id in input_vuln_ids:
        if vuln_id in engine_output_vuln_ids:
            output.append(
                parse_agent_morpheus_engine_output(vuln_id=vuln_id,
                                                   checklist_results=result.checklist_results[vuln_id],
                                                   summary=result.final_summaries[vuln_id],
                                                   justification=result.justifications[vuln_id]))
        else:
            # Handle skipped vulnerabilities based on skip reason
            skip_reason = skip_reasons[vuln_id]
            if skip_reason == LLMEngineSkipReason.NO_SBOM_PACKAGES:
                output.append(build_no_sbom_output(vuln_id))
            elif skip_reason == LLMEngineSkipReason.MISSING_SOURCE:
                output.append(build_missing_source_action_output(vuln_id))
            elif skip_reason == LLMEngineSkipReason.INSUFFICIENT_INTEL:
                output.append(build_deficient_intel_output(vuln_id))
            elif skip_reason == LLMEngineSkipReason.NO_VULNERABLE_PACKAGES:
                output.append(build_no_vuln_packages_output(vuln_id))
            elif skip_reason is None:
                assert False, f"Vuln ID {vuln_id} was not skipped but there is no LLM Engine output."
            else:
                assert False, f"Vuln ID {vuln_id} has an unknown skip reason: {skip_reason}"

    for out in output:
        logger.info("Vulnerability '%s' affected status: %s. Label: %s",
                    out.vuln_id,
                    out.justification.status,
                    out.justification.label)

    return AgentMorpheusOutput(input=message.input, info=message.info, output=output)
