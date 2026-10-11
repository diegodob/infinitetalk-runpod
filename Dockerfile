# InfiniteTalk en RunPod — imagen lista para correr
#
# Base: la imagen oficial de RunPod con PyTorch 2.8.0 + CUDA 12.8 (x86_64, la misma
# arquitectura que usan los pods). Sobre eso: las dependencias con versiones fijas,
# el repositorio de InfiniteTalk en un commit puntual y los parches de compatibilidad.
#
# Los modelos (38 GB) NO van adentro de la imagen: se bajan con scripts/setup_weights.sh
# en unos 5 minutos, o se dejan en un volumen de red de RunPod.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git aria2 ninja-build ffmpeg rsync ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /tmp/constraints.txt

# Entorno virtual con acceso a los paquetes del sistema (asi hereda torch/torchvision
# ya instalados en la imagen base y no se bajan de nuevo).
RUN python3 -m venv --system-site-packages /opt/venv \
 && pip install --no-cache-dir -q -U pip setuptools wheel \
 && pip install --no-cache-dir -q -c /tmp/constraints.txt \
        "diffusers==0.31.0" "transformers==4.51.3" "accelerate==0.34.2" \
        soundfile librosa imageio imageio-ffmpeg einops decord pyloudnorm \
        sentencepiece ftfy "protobuf==7.36.2" "huggingface_hub[cli]" \
        easydict regex tqdm safetensors scipy modelscope dashscope tokenizers \
        opencv-python-headless "optimum-quanto==0.2.6" scenedetect \
        "moviepy==1.0.3" scikit-image loguru av ninja \
 && pip install --no-cache-dir -q --no-deps "xformers==0.0.32.post1"

# El repositorio, en el commit exacto con el que se probo todo esto.
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /opt/italk \
 && cd /opt/italk && git checkout 50aa0a94184315407a991ae804d9b58d6d311ba8

COPY patches/ /opt/patches/
# Chequeos que SI se pueden hacer en el build (sin placa de video):
#  - los parches se aplican
#  - los archivos parcheados compilan (atrapa errores de sintaxis)
#  - el stub de xfuser responde
# NO se puede hacer "import wan" aca: wan/modules/t5.py llama a torch.cuda.current_device()
# al importarse, y el constructor de Docker no tiene GPU -> "Found no NVIDIA driver".
# Esa verificacion se hace al arrancar el pod (justo despues, con placa).
RUN python3 /opt/patches/apply_patches.py /opt/italk \
 && cd /opt/italk \
 && python3 -m py_compile wan/multitalk.py wan/modules/attention.py wan/modules/t5.py \
      wan/utils/multitalk_utils.py generate_infinitetalk.py \
 && python3 -c "import xfuser.core.distributed as d; assert d.get_world_size() == 1; assert d.get_rank() == 0; print('stub de xfuser ok')" \
 && python3 -c "exec('from xfuser.core.distributed import *'); print('import * del stub ok')"

WORKDIR /opt/italk
CMD ["bash"]
