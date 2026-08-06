# =============================================================================
# Docker image untuk pipeline Subtitle Detect + SAM2 Refine + LaMa Inpaint.
#
# Isi image ini = persis step 2-10 dari workflow original (install system
# deps, install detection deps, clone+patch LaMa, install LaMa deps dengan
# versi-pin yang sama, download checkpoint big-lama) — HANYA dipindah dari
# "jalan tiap job" jadi "jalan sekali saat build image".
#
# Image ini di-build & di-push sekali oleh workflow build-image.yml, lalu
# dipakai berulang oleh semua job matrix di pipeline.yml tanpa install ulang.
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/opt
ENV PYTHONPATH=/opt/lama_repo
ENV SAM2_WEIGHTS_PATH=/opt/models/sam2_l.pt

# -----------------------------------------------------------------------
# 1) System dependencies
#    (identik dengan step "Install System Dependencies" workflow original)
# -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 unzip ffmpeg curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# -----------------------------------------------------------------------
# 2) Detection/SAM2 dependencies
#    (identik dengan step "Install Detection/SAM2 Dependencies")
# -----------------------------------------------------------------------
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install easyocr opencv-python-headless pillow numpy ultralytics

# -----------------------------------------------------------------------
# 3) Clone official LaMa repo + patch torch.load
#    (identik dengan step "Clone official LaMa repository")
# -----------------------------------------------------------------------
RUN git clone https://github.com/advimman/lama.git /opt/lama_repo && \
    sed -i \
      "s/torch.load(path, map_location=map_location)/torch.load(path, map_location=map_location, weights_only=False)/" \
      /opt/lama_repo/saicinpainting/training/trainers/__init__.py && \
    grep -n "torch.load" /opt/lama_repo/saicinpainting/training/trainers/__init__.py

# -----------------------------------------------------------------------
# 4) LaMa's own dependencies, dengan version-pin yang sama seperti original
#    (identik dengan step "Install LaMa Dependencies")
# -----------------------------------------------------------------------
RUN sed -i -E 's/==[^[:space:]]+//' /opt/lama_repo/requirements.txt && \
    python3 -m pip install -r /opt/lama_repo/requirements.txt && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall && \
    python3 -m pip install wldhx.yadisk-direct && \
    python3 -m pip install "albumentations==0.5.2" "imgaug>=0.4.0" && \
    python3 -m pip install "kornia==0.5.0" && \
    python3 -m pip install "numpy<2" "opencv-python-headless<5"

# -----------------------------------------------------------------------
# 5) Download checkpoint big-lama, dibakar langsung ke dalam image
#    (identik dengan step "Download big-lama checkpoint", mirror HF -> Yandex)
# -----------------------------------------------------------------------
WORKDIR /opt
RUN ( curl -fL --connect-timeout 15 --max-time 180 --retry 3 --retry-delay 5 \
        -o big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip \
      && echo "Berhasil unduh dari Hugging Face mirror." ) \
    || ( echo "Mirror Hugging Face gagal, coba Yandex Disk..." \
         && DIRECT_URL=$(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) \
         && curl -fL --connect-timeout 15 --max-time 180 --retry 3 --retry-delay 5 \
              -o big-lama.zip "$DIRECT_URL" ) \
    && unzip -o big-lama.zip -d /opt \
    && rm big-lama.zip \
    && ls -la /opt/big-lama

# -----------------------------------------------------------------------
# 6) Pre-download bobot SAM2 & EasyOCR sekali di build time, supaya job
#    matrix tidak perlu download model lagi tiap kali jalan.
# -----------------------------------------------------------------------
RUN mkdir -p /opt/models && cd /opt/models && \
    python3 -c "from ultralytics import SAM; SAM('sam2_l.pt')"

RUN python3 -c "import easyocr; easyocr.Reader(['en','id'], gpu=False)"

# -----------------------------------------------------------------------
# 7) Salin script pipeline (detect_and_mask.py, prepare_chunks.py) ke image
# -----------------------------------------------------------------------
COPY scripts/ /workspace/scripts/

WORKDIR /workspace
