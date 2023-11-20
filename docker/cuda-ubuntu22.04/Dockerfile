# Stage 1: Build Environment Setup
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as builder

RUN apt-get update -y && apt-get install -y wget curl git tar bzip2 && rm -rf /var/lib/apt/lists/*

# Create a user
ENV APPUSER="appuser"
ENV HOME=/home/$APPUSER
RUN useradd -m -u 1000 $APPUSER
USER $APPUSER
WORKDIR $HOME

ENV ENV_NAME="diffdock-pocket"

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
ENV PATH=$HOME/bin:$HOME/.local/bin:$PATH

# Copy and create Conda environment
ENV ENV_FILE_NAME=environment.yml
COPY --chown=$APPUSER:$APPUSER ./$ENV_FILE_NAME .
RUN ~/bin/micromamba env create --file $ENV_FILE_NAME && ~/bin/micromamba clean -afy --quiet

# Copy application code
COPY --chown=$APPUSER:$APPUSER . $HOME/DiffDock-Pocket

# Stage 2: Runtime Environment
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Create user and setup environment
ENV APPUSER="appuser"
ENV HOME=/home/$APPUSER
RUN useradd -m -u 1000 $APPUSER
USER $APPUSER
WORKDIR $HOME

ENV ENV_NAME="diffdock-pocket"

# Copy the Conda environment and application code from the builder stage
COPY --from=builder --chown=$APPUSER:$APPUSER $HOME/micromamba $HOME/micromamba
COPY --from=builder --chown=$APPUSER:$APPUSER $HOME/bin $HOME/bin
COPY --from=builder --chown=$APPUSER:$APPUSER $HOME/DiffDock-Pocket $HOME/DiffDock-Pocket
WORKDIR $HOME/DiffDock-Pocket

# Set the environment variables
ENV MAMBA_ROOT_PREFIX=$HOME/micromamba
ENV PATH=$HOME/bin:$HOME/.local/bin:$PATH
RUN micromamba shell init -s bash --root-prefix $MAMBA_ROOT_PREFIX

# Expose ports for streamlit and gradio
EXPOSE 7860 8501

# Default command
CMD ["sh", "-c", "micromamba run -n ${ENV_NAME} python utils/print_device.py"]
