FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    liblapacke-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip \
    nodejs \
    npm \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    opencv-python \
    tensorflow \
    scikit-learn \
    pandas \
    mediapipe \
    supabase \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    requests

# Clone OpenPose repository
WORKDIR /opt
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Download models
WORKDIR /opt/openpose/models
RUN ./getModels.sh

# Build OpenPose
WORKDIR /opt/openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j`nproc`

# Set up app directory
WORKDIR /app
COPY . .

# Install Node.js dependencies
RUN npm install

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Expose API port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]