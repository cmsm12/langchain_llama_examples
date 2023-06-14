FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ninja-build \
    build-essential
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn
RUN pip install llama-cpp-python

# clear apt cache
RUN rm -rf /var/lib/apt/lists/*

# Copy prediction server code to image
COPY ./app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]