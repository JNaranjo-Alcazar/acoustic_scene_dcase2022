FROM nvidia/cuda:11.3.0-base 
FROM tensorflow/tensorflow:latest-gpu

#Copy sources
WORKDIR /app
COPY requirements.txt .

# Install pip requirements
ARG PYTHON=python3.8
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y  \ 
    libsndfile-dev \
    ${PYTHON} \
    ${PYTHON}-dev \
    ${PYTHON}-distutils \
    curl \
    git \
    build-essential

    
# Install python requirement
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html