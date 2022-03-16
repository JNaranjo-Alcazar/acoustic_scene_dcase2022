FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

WORKDIR /app

COPY requirements.txt .

RUN  git clone https://github.com/google-research/leaf-audio.git
RUN cd leaf-audio && pip3 install -e .

RUN pip3 install lambda-networks

RUN pip3 install torch-summary

RUN pip3 install nnAudio==0.3.1

RUN pip3 install pytorch-lightning==1.5.10