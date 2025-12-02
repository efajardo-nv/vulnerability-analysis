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

import argparse
import json
import os

import matplotlib.pyplot as plt
"""
Box-and-whisker plot utility for evaluator outputs.

Example usage:

    python3 src/vuln_analysis/eval/visualizations/box_and_whisker_plot.py \
      .tmp/evaluators/llama8b-experiment/status_accuracy_output.json \
      .tmp/evaluators/llama70b-experiment/status_accuracy_output.json \
      --title "Status Accuracy: llama8b vs llama70b" \
      --save src/vuln_analysis/eval/visualizations/plot.png \
      --labels llama8b llama70b
"""


def load_box_values(path: str) -> tuple[list[float], str]:
    """Load average_score as [mean, min, q1, median, q3, max] from accuracy evaluator output JSON.
    Returns (values, label) where label is the input file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    avg = data.get("average_score")
    if not isinstance(avg, dict):
        raise ValueError(f"File '{path}' does not contain pandas describe() format. Expecting dict in 'average_score'.")

    # Extract values from pandas describe() format: [mean, min, q1, median, q3, max]
    values = [
        float(avg.get("mean", 0)),
        float(avg.get("min", 0)),
        float(avg.get("25%", 0)),
        float(avg.get("50%", 0)),
        float(avg.get("75%", 0)),
        float(avg.get("max", 0))
    ]

    label = os.path.splitext(os.path.basename(path))[0]
    return values, label


def main() -> int:
    parser = argparse.ArgumentParser(description=(
        "Plot box-and-whisker charts from evaluator output JSON files (average_score = [mean, min, q1, median, q3, max])."
    ))
    parser.add_argument("files", nargs="+", help="Paths to output files")
    parser.add_argument("--title", default="Accuracy Distribution Across Runs", help="Chart title.")
    parser.add_argument("--save",
                        metavar="OUTPUT_PATH",
                        help="If provided, save the figure to this path instead of displaying it.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI when saving (default: 160).")
    parser.add_argument(
        "--labels",
        nargs='*',
        help="Optional custom x-axis labels (one per file, in order). If omitted, .json file names are used.")

    args = parser.parse_args()

    # Load all files
    stats_values: list[dict] = []
    means_overlay: list[tuple[float, float]] = []
    labels: list[str] = []

    for idx, path in enumerate(args.files):
        values, default_label = load_box_values(path)
        mean_value, min_value, q1, median, q3, max_value = values

        # Matplotlib boxplot stats dictionary
        stats_values.append({'whislo': min_value, 'q1': q1, 'med': median, 'q3': q3, 'whishi': max_value, 'fliers': []})
        if args.labels and len(args.labels) == len(args.files):
            labels.append(args.labels[idx])
        else:
            labels.append(default_label)
        means_overlay.append((len(labels), mean_value))

    if not stats_values:
        print("No valid files provided.")
        return 1

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(stats_values)), 5))

    # Create boxplot from precomputed stats
    bp = ax.bxp(stats_values, showfliers=False)

    # Overlay means as black diamonds
    for x_pos, mean_value in means_overlay:
        ax.plot(x_pos, mean_value, marker='D', color='black', linestyle='None', label='_nolegend_')

    ax.set_title(args.title)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend entry for mean marker
    ax.plot([], [], marker='D', color='black', linestyle='None', label='Mean')
    ax.legend()

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
