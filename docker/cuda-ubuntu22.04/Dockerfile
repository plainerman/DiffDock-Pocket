FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get install -y sudo wget curl nano git && rm -rf /var/lib/apt/lists/*

# Create a user
ENV APPUSER="appuser"
ENV HOME=/home/$APPUSER
RUN useradd -m -u 1000 $APPUSER

USER $APPUSER
WORKDIR $HOME

RUN mkdir -p ~/.cache/torch/hub/checkpoints
# Download ESM models
ENV ESM_MODEL="esm2_t33_650M_UR50D"
ENV CONTACT_MODEL="$ESM_MODEL-contact-regression"
RUN curl -o ~/.cache/torch/hub/checkpoints/$ESM_MODEL.pt https://dl.fbaipublicfiles.com/fair-esm/models/$ESM_MODEL.pt
RUN curl -o ~/.cache/torch/hub/checkpoints/$CONTACT_MODEL.pt https://dl.fbaipublicfiles.com/fair-esm/regression/$CONTACT_MODEL.pt

# Build is significantly faster with micromamba, rather than conda
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba

# Set the environment variables
ENV MAMBA_ROOT_PREFIX=$HOME/micromamba
ENV PATH=$HOME/bin:$HOME/.local/bin:$PATH

RUN ~/bin/micromamba shell init -s bash --root-prefix $MAMBA_ROOT_PREFIX  # this writes to your .bashrc file
RUN ~/bin/micromamba --version

# Create conda environment first
# This is time-consuming and don't want to reproduce it often
ENV ENV_FILE_NAME=environment.yml
ENV ENV_NAME="diffdock-pocket"
COPY --chown=$APPUSER:$APPUSER ./$ENV_FILE_NAME .

RUN ~/bin/micromamba env create --file $ENV_FILE_NAME && ~/bin/micromamba clean -afy

COPY --chown=$APPUSER:$APPUSER . $HOME/DiffDock-Pocket

WORKDIR $HOME/DiffDock-Pocket

# Expose default streamlit and gradio ports
EXPOSE 7860 8501

# Default command just prints the device (CPU or GPU), also ensure using correct python
CMD ["sh", "-c", "micromamba run -n ${ENV_NAME} python utils/print_device.py"]
