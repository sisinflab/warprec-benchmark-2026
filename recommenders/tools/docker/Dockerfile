# syntax=docker/dockerfile:1

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Stage build order depending on the compute:
#     Compute Stage (CPU/GPU) -> Dependencies Stage -> Final Stage
#####################################################################
# Valid computes: cpu, gpu
ARG COMPUTE="cpu"


#####################################################################
# Compute Stage - CPU
# Choose an appropriate CPU compute image
#####################################################################
# * [buildpack-deps:24.04](https://github.com/docker-library/buildpack-deps/blob/master/ubuntu/noble/Dockerfile)
#   + [Created on 2024-08-17](https://hub.docker.com/layers/library/buildpack-deps/noble/images/sha256-dbfee7e7ee2340b0d6567efd3a8a9281ce45ee78598485b4d7a7f09fe641811a)
FROM buildpack-deps@sha256:dbfee7e7ee2340b0d6567efd3a8a9281ce45ee78598485b4d7a7f09fe641811a AS cpu


#####################################################################
# Compute Stage - GPU
# Choose an appropriate GPU compute image
#####################################################################
# * [nvidia/cuda:12.6.1-devel-ubuntu24.04](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.1/ubuntu2404/devel/Dockerfile)
#   + [Created on 2024-09-13](https://hub.docker.com/layers/nvidia/cuda/12.6.1-devel-ubuntu24.04/images/sha256-bfc293f21611f3c47a3442cf6516ebfe99d529926a4bef4bc389ef02fd038800)
# * See also [AML GPU Base Image](https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04)
FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu24.04@sha256:bfc293f21611f3c47a3442cf6516ebfe99d529926a4bef4bc389ef02fd038800 AS gpu

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y wget git && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


#####################################################################
# Dependencies Stage
# Set up all dependencies.  This Stage is used by dev containers,
# because editable installation is required.
#####################################################################
FROM ${COMPUTE} AS deps

# Valid versions: 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /root
USER root:root

SHELL ["/bin/bash", "-c"]

# Install Python and uv
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create virtual environment with uv
RUN uv venv /root/.venvs/Recommenders --python python${PYTHON_VERSION}

# Activate virtual environment by default
ENV VIRTUAL_ENV="/root/.venvs/Recommenders"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"


#####################################################################
# Final Stage
# Install Recommenders
#####################################################################
FROM deps AS final

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

# Git ref of Recommenders to install: main, staging, etc.
# Empty value ("") indicates editable installation of current clone
ARG GIT_REF="main"

ARG JDK_VERSION="21"

ARG RECO_DIR="/root/Recommenders"

# Install Java (Temurin) for Spark support if needed
RUN if [[ "${EXTRAS}" =~ spark ]]; then \
        apt-get update && \
        apt-get install -y wget apt-transport-https gpg && \
        wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(. /etc/os-release && echo $VERSION_CODENAME) main" | tee /etc/apt/sources.list.d/adoptium.list && \
        apt-get update && \
        apt-get install -y temurin-${JDK_VERSION}-jdk && \
        apt-get clean -y && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Set JAVA_HOME for Spark
ENV JAVA_HOME="/usr/lib/jvm/temurin-${JDK_VERSION}-jdk-amd64"
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy Recommenders into the image
COPY ./ ${RECO_DIR}

# Install Recommenders and its dependencies using uv
RUN if [ -z "${GIT_REF}" ]; then \
        uv pip install ${RECO_DIR}${EXTRAS}; \
    else \
        uv pip install recommenders${EXTRAS}@git+https://github.com/recommenders-team/recommenders.git@${GIT_REF}; \
    fi && \
    uv pip install jupyter ipykernel && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

# Setup Jupyter notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.allow_origin='*'", "--IdentityProvider.token=''"]
