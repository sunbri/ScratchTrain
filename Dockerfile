# Python Image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy data
COPY data/ ./data/

# Copy dependency files
COPY pyproject.toml uv.lock ./
# Install dependencies using uv
RUN uv sync

# Copy usually modified files
COPY checkpoints/ ./checkpoints/
COPY output/ ./output/
COPY src/ ./src/

# Set default command
CMD ["sh", "-c", "uv run python src/run.py"]