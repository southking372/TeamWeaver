# Third-Party Notices and Attribution

TeamWeaver extends [PARTNR / habitat-llm](https://github.com/facebookresearch/partnr-planner) (Meta Platforms, Inc.). This document lists upstream attribution markers in the codebase for copyright compliance.

## 1. Meta / PARTNR file header (135 Python files)

Many files retain the upstream MIT license header from PARTNR:

```text
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
```

Some files append a trailing period on the last line; the meaning is the same.

### Directories that are primarily Meta-derived (keep headers unchanged)

| Directory | Description |
|-----------|-------------|
| `world_model/` | World graph, entities, dynamic/privileged graphs |
| `tools/` | Perception tools and motor skills |
| `utils/` | Grammar, geometry, sim helpers, semantic constants |
| `llm/` | Llama, OpenAI chat, HF model wrappers |
| `perception/` | Perception stack |
| `sims/` | Habitat sim metadata and collaboration sim |
| `agent/` | Agent definition |
| `evaluation/centralized_evaluation_runner.py` | Base PARTNR evaluation runner |
| `evaluation/decentralized_evaluation_runner.py` | Decentralized evaluation |
| `evaluation/evaluation_runner.py` | Evaluation base classes |
| `examples/` (except TeamWeaver-specific demos) | PARTNR example scripts |
| `finetuning/` | Trace dataset and trainer |
| `tests/` | PARTNR unit tests |
| `concept_graphs/` | Concept graph utilities |
| `conf/` | Hydra configs (PARTNR baseline layouts) |
| `planner/planner.py`, `llm_planner.py`, `rag.py`, etc. | Core PARTNR planners |

## 2. Import namespace: `habitat_llm.*`

Runtime imports still use the upstream package name `habitat_llm` (e.g. `from habitat_llm.agent.env import EnvironmentInterface`). This is the **Python module namespace** from PARTNR, not a separate copyright notice, but it indicates code paths that depend on upstream APIs.

TeamWeaver-specific modules (typically **without** Meta headers) include:

| Directory / file | Origin |
|------------------|--------|
| `planner/HRCS/` | TeamWeaver HRCS / MIQP integration |
| `planner/human_modeling/` | Human modeling system |
| `planner/miqp_planner/` | Standalone MIQP task planner |
| `planner/perception_connector.py` | Perception–MIQP connector |
| `planner/hrcs_llm_planner.py` | HRCS LLM planner |
| `planner/llm_planner_miqp.py` | LLM + MIQP hybrid (extends Meta `LLMPlanner`) |
| `evaluation/hrcs_centralized_evaluation_runner*.py` | HRCS evaluation |
| `evaluation/coherence_evaluator.py` | Transparency metrics |
| `evaluation/truthfulness_evaluator.py` | Truthfulness metrics |
| `evaluation/hallucination_experiment.py` | Hallucination experiments |

Files that **both** carry Meta headers **and** TeamWeaver extensions (e.g. `llm_planner_miqp.py`, `coherence_evaluator.py`) should **retain** the Meta header and document TeamWeaver changes in commit history / this file.

## 3. Documentation and config references

| Reference | Where | Action |
|-----------|-------|--------|
| `PARTNR` / `partnr-planner` | `README.md`, `world_model/README.md` | Keep attribution in Acknowledgments |
| `facebookresearch/partnr-planner` | README install links | Keep upstream link |
| `Meta's Habitat platform` | README Acknowledgments | Keep |
| `habitat-llm` in comments | Various planners, configs | Informational; optional to rename in docs only |

## 4. Recommended copyright layout for TeamWeaver

1. **Do not remove** Meta headers from files that originated in PARTNR.
2. **Add** a root `LICENSE` (MIT for TeamWeaver-authored parts) and keep upstream MIT terms for Meta-derived files.
3. **Add** SPDX or a short header only on **new** TeamWeaver files, e.g.:

   ```text
   # Copyright (c) 2025-2026 TeamWeaver contributors.
   # SPDX-License-Identifier: MIT
   ```

4. **README Acknowledgments** (already present): cite PARTNR and Habitat.
5. **Optional**: For heavily modified Meta files, add after the Meta block:

   ```text
   # Modifications Copyright (c) 2025-2026 TeamWeaver contributors.
   ```

## 5. External dependencies (not shipped as source here)

- **Habitat-Sim / Habitat-Lab** — Meta / FAIR ecosystem
- **PARTNR dataset and checkpoints** — Meta, downloaded separately per PARTNR install guide
- **Llama models** — Meta (via HuggingFace)
- **OpenAI / Azure OpenAI APIs** — optional inference backend

## 6. Archive code

`planner/llm_planner_bak/` is internal development history, not imported by the package. It may be removed before public release or kept out of distribution; it is not part of upstream PARTNR.
