FROM python:3.9-slim

WORKDIR /workspace/DupNAS-AE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        patchelf \
        curl \
        wget \
        unzip \
        libusb-1.0-0 \
        libglib2.0-0 \
        usbutils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_base.txt .

# Install the required CUDA 11.5 PyTorch build
RUN python3.9 -m pip install --no-cache-dir \
        torch==1.11.0+cu115 \
        torchvision==0.12.0+cu115 \
        torchaudio==0.11.0 \
        --extra-index-url https://download.pytorch.org/whl/cu115

# Install the remaining dependencies without reinstalling PyTorch packages
RUN grep -vE '^(torch|torchvision|torchaudio)==' \
        requirements_base.txt > requirements-final.txt && \
    python3.9 -m pip install --no-cache-dir \
        -r requirements-final.txt

# Clear the executable-stack requirement from libtorch_cpu.so
RUN LIB="$(find /usr/local/lib/python3.9/site-packages/torch \
        -name 'libtorch_cpu.so' -print -quit)" && \
    test -n "$LIB" && \
    echo "Clearing executable-stack flag from: $LIB" && \
    patchelf --clear-execstack "$LIB"

# Verify that PyTorch can be imported during the build
RUN python3.9 -c "import torch; print('PyTorch version:', torch.__version__)"

COPY . .

CMD ["bash"]