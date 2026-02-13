# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \ 
    cmake \  
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    alsa-oss \
    pulseaudio \
    v4l-utils \
    ffmpeg \  
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install PyTorch (CPU version for smaller image size on RPi)
# For CUDA support, use: https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
ADD . /app

# Create ALSA configuration files
RUN echo "pcm.!default {\n type hw\n card 0\n}\nctl.!default {\n type hw\n card 0\n}" > /etc/asound.conf
RUN echo "@hooks [\n{\n func load\n files [\n\"/etc/asound.conf\"\n ]\n errors false\n}\n]" > /usr/share/alsa/alsa.conf

# Pre-download PyTorch models to avoid runtime downloads
# FaceNet and YOLO models will be cached here
RUN python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; \
    MTCNN(); \
    InceptionResnetV1(pretrained='vggface2')" || true

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME=World
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Run app.py when the container launches
CMD ["python", "manage.py", "runserver", "0.0.0.0:80"]
