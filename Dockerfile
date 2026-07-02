FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.0.0
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY main.py .
COPY worker_api.py .
COPY best.pt ./best.pt
COPY model/v4 ./model/v4

EXPOSE 8000

CMD ["python3", "main.py"]