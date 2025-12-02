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

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from vuln_analysis.data_models.input import ImageInfoInput
from vuln_analysis.utils.justification_parser import JustificationParser
from vuln_analysis.utils.string_utils import is_valid_cve_id
from vuln_analysis.utils.string_utils import is_valid_ghsa_id

# Use label/status mappings from JustificationParser
ALLOWED_LABELS: tuple[str, ...] = tuple(JustificationParser.JUSTIFICATION_TO_AFFECTED_STATUS_MAP.keys())

# Map justification status (TRUE/FALSE/UNKNOWN) to eval dataset statuses (AFFECTED/NOT AFFECTED/UNKNOWN)
_JUST_STATUS_TO_EVAL_STATUS: dict[str, str] = {
    "TRUE": "AFFECTED",
    "FALSE": "NOT AFFECTED",
    "UNKNOWN": "UNKNOWN",
}

ALLOWED_STATUSES: tuple[str, ...] = tuple(_JUST_STATUS_TO_EVAL_STATUS.values())


class GroundTruthItem(BaseModel):
    vuln_id: str
    status: str
    label: str

    @field_validator("vuln_id")
    @classmethod
    def validate_vuln_id(cls, v: str) -> str:
        if not (is_valid_cve_id(v) or is_valid_ghsa_id(v)):
            raise ValueError(f"{v} is not a valid CVE ID or GHSA ID.")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in ALLOWED_LABELS:
            raise ValueError(f"{v} is not a valid justification label. Allowed: {sorted(ALLOWED_LABELS)}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ALLOWED_STATUSES:
            raise ValueError(f"{v} is not a valid status. Allowed: {sorted(ALLOWED_STATUSES)}")
        return v

    @model_validator(mode="after")
    def validate_label_status_consistency(self):
        # Expected status from label via mapping from JustificationParser
        expected_status = JustificationParser.JUSTIFICATION_TO_AFFECTED_STATUS_MAP[self.label]
        expected_status_transformed = _JUST_STATUS_TO_EVAL_STATUS[expected_status]

        if self.status != expected_status_transformed:
            raise ValueError(
                f"Label/status mismatch: label '{self.label}' implies '{expected_status_transformed}' but got '{self.status}'."
            )

        return self


class AgentMorpheusEvalInputItem(BaseModel):
    container_image: ImageInfoInput
    ground_truth: list[GroundTruthItem]


class AgentMorpheusEvalDataset(BaseModel):
    dataset_id: str
    dataset_description: str | None = None
    containers: dict[str, AgentMorpheusEvalInputItem]
