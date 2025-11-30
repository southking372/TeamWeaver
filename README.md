# TeamWeaver: Multi-Robot Collaborative Planning Framework

## 📋 Overview

TeamWeaver is a multi-robot collaborative task planning framework that integrates Large Language Models (LLMs) with Mixed-Integer Quadratic Programming (MIQP). The framework combines the semantic understanding capabilities of LLMs with the mathematical optimization power of MIQP to enable intelligent task allocation and execution for heterogeneous multi-robot teams in dynamic environments. This project extends the [PARTNR benchmark](https://github.com/facebookresearch/partnr-planner) with our LMs-MIQP hybrid planning algorithm.

## ✨ Key Features

- **Hybrid Planning Architecture**: Combines LLM semantic reasoning with MIQP constraint optimization for interpretable and optimal task allocation
- **Heterogeneous Robot Support**: Supports various robot types (mobile platforms, manipulators, humanoid robots) working collaboratively
- **Dynamic Environment Adaptation**: Handles dynamic environments, communication failures, and human interventions
- **Transparency Metrics**: Provides quantifiable metrics including Semantic Transparency ($S_{TL}$) and Execution Transparency ($S_{TE}$)
- **Modular Design**: Adopts Perception-Plan-Skill architecture for easy extension and maintenance

## 🏗️ System Architecture

### Core Components

1. **Planner**
   - `llm_planner_miqp.py`: LLM-MIQP hybrid planner
   - `llm_planner.py`: Pure LLM planner
   - `HRCS/`: Human-Robot Collaboration System modules

2. **World Model**
   - `world_graph.py`: Privileged world graph (based on full observations)
   - `dynamic_world_graph.py`: Dynamic world graph (based on partial observations)

3. **LLM Interface**
   - Support for Llama 2.X/3.X (base and instruction-tuned versions)
   - Support for OpenAI Chat models (via Azure OpenAI)
   - Support for multimodal LLMs

4. **Tool System**
   - **Perception Tools**: Object finding, room querying, map querying, etc.
   - **Motor Skills**: Navigation, exploration, pick, place, rearrange, etc.

5. **Evaluation System**
   - Task execution metrics: Success Rate (SR), Percent Complete (PC), Simulation Steps (SS)
   - Transparency metrics: Semantic Transparency ($S_{TL}$), Execution Transparency ($S_{TE}$)

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Habitat-Sim (see [PARTNR installation guide](https://github.com/facebookresearch/partnr-planner/blob/main/INSTALLATION.md))
- PyTorch
- Supported LLM models (Llama or OpenAI API)

### Installation

Please refer to the [PARTNR installation guide](https://github.com/facebookresearch/partnr-planner/blob/main/INSTALLATION.md) for detailed setup instructions. The installation process includes:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd teamweaver
```

2. **Install Habitat-Sim**: Follow the Habitat-Sim installation instructions from the PARTNR repository, which typically involves:
   - Installing system dependencies
   - Building Habitat-Sim from source or using pre-built binaries
   - Setting up the Habitat-Lab environment

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download datasets**: Download the PARTNR dataset as specified in the PARTNR repository

5. **Download skill checkpoints** (if using neural network skills): Refer to the PARTNR installation guide for skill checkpoint downloads

### Configure LLM

#### Using Llama Models (HuggingFace)

You can use HuggingFace models directly by setting the model path:

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.planner.plan_config.llm.inference_mode=hf \
    evaluation.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

Or use a local model path:

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.planner.plan_config.llm.inference_mode=hf \
    evaluation.planner.plan_config.llm.generation_params.engine="path/to/your/llama/model"
```

#### Using OpenAI API

Set environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_ENDPOINT="your-endpoint"
```

Then run:

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    llm@evaluation.planner.plan_config.llm=openai_chat
```

### Run Examples

#### Centralized Multi-Agent with LLM-MIQP Planner

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.planner.plan_config.llm.inference_mode=hf \
    evaluation.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

#### Decentralized Multi-Agent

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

#### Single Agent

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

#### Custom Instructions

You can also run custom instructions:

```bash
python -m teamweaver.examples.planner_demo \
    --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    instruction="<CUSTOM_INSTRUCTION>" \
    mode='cli' \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

## 📁 Directory Structure

```
teamweaver/
├── agent/              # Agent interface and environment interaction
├── planner/            # Planner implementations
│   ├── llm_planner_miqp.py    # LLM-MIQP hybrid planner (TeamWeaver core)
│   ├── HRCS/          # Human-Robot Collaboration System modules
│   └── miqp_planner/   # MIQP planner core
├── llm/               # LLM interface and configuration
├── world_model/       # World graph representation and management
├── tools/             # Perception tools and motor skills
├── evaluation/        # Evaluation system and metrics
├── examples/          # Example scripts
├── conf/              # Configuration files (Hydra)
└── tests/             # Unit tests
```

## 🔧 Configuration

The project uses Hydra for configuration management. Main configuration files are located in `conf/`:

- `conf/planner/`: Planner configurations
- `conf/llm/`: LLM model configurations
- `conf/tools/`: Tool configurations
- `conf/instruct/`: Instruction template configurations

## 📊 Evaluation Metrics

### Task Execution Metrics
- **Success Rate (SR)**: Percentage of successfully completed tasks
- **Percent Complete (PC)**: Percentage of task completion
- **Simulation Steps (SS)**: Number of simulation steps required to complete tasks
- **Replanning Count (RC)**: Number of replanning events during task execution

### Transparency Metrics
- **Semantic Transparency ($S_{TL}$)**: Alignment between LLM-generated semantics and execution plans
- **Execution Transparency ($S_{TE}$)**: Consistency between execution process and planning decisions

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_planner.py
pytest tests/test_evaluation.py
```

## 📚 Documentation

Detailed documentation for each module:

- [LLM Configuration](llm/README.md)
- [World Model](world_model/README.md)
- [HRCS Plan Module](planner/HRCS/plan_module/README.md)
- [Motor Skills](tools/motor_skills/README.md)

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📊 Calculating Results

You can check the progress and results of your runs using:

```bash
python scripts/read_results.py <output_dir>/<dataset_name>
```

The default `output_dir` is `outputs/teamweaver/<timestamp>-<dataset_name>`. This can be overridden using the `paths.results_dir` and `evaluation.output_dir` configuration elements.

## 🙏 Acknowledgments

This project is built upon Meta's Habitat platform and the [PARTNR benchmark](https://github.com/facebookresearch/partnr-planner). We extend the PARTNR framework with our LMs-MIQP hybrid planning algorithm for improved task allocation and execution in heterogeneous multi-robot systems.

