# Justfile for managing the project tasks
python_version := "3.10.19"

# Create a virtual environment and install dependencies
install:
    if [ ! -d .venv ]; then uv venv --python {{python_version}}; fi
    uv sync --all-groups

# Install only production dependencies (no dev dependencies)
install-prod:
    if [ ! -d .venv ]; then uv venv --python {{python_version}}; fi
    uv sync --no-dev

# Format the code with black
format:
    uv run black syncstream/ tests/

# Lint the code with bandit
lint:
    uv run bandit -r syncstream/

# Run all tests (Unit + Integration)
test-all:
    uv run pytest tests/

# Run only integration tests
test-integration:
    uv run pytest tests/test_pipeline_integration.py

# Run only unit tests
test-unit:
    uv run pytest tests/test_pipeline_unit.py

# Run the main script with a specified input file
run input="data/kitti.mcap":
    uv run python main.py --input {{input}}

# Run the dataset script to generate the output sample data for validation
run-dataset:
    uv run python syncstream/dataset.py

# Create a .gif visualization of the output data
viz input="output/":
    uv run python visualizer.py --dir {{input}}

# Setup the environment and install pre-commit hooks
setup-dev: install
    uv run pre-commit install

# Manually run all hooks against all files
check:
    uv run pre-commit run --all-files

# Run the pipeline inside a Docker container
docker:
    docker compose run --rm converter
