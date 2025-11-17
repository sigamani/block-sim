# Block-Sim: Vidur Large Language Model Serving Simulator

Block-Sim is a simulation framework for Vidur, designed to model and analyze the performance of large language model (LLM) serving systems. It includes components for request generation, scheduling, execution time prediction, and metrics collection.

## Features

- **Request Generation**: Synthetic and trace-based request generators for realistic workloads.
- **Global and Replica Schedulers**: Algorithms for distributing requests across model replicas.
- **Execution Time Predictors**: Models to estimate inference times based on request characteristics.
- **Metrics and Analysis**: Comprehensive metrics for throughput, latency, and resource utilization.
- **Experiment Scripts**: Tools for running and analyzing simulation experiments.

## Installation

- Python 3.10+
- Install dependencies: `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`

## Usage

- Key scripts: `sh block/exp/setup.sh`, `python block/exp/generate_config.py ...`
- Run simulation: `python vidur/main.py` (with appropriate config)
- For vLLM integration (optional for testing): Use Docker image `michaelsigamani/block:v1` with `python3 -m vllm.entrypoints.api_server --model microsoft/phi-2 --port 8000 --trust-remote-code`

## Project Structure

- `block/`: Core simulation logic
- `vidur/`: Vidur-specific components
- `data/`: Datasets for training and testing
- `experiments_analysis/`: Analysis tools

## License

[Add appropriate license]