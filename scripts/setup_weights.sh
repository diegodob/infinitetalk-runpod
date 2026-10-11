#!/bin/bash
# Baja los modelos que InfiniteTalk necesita (unos 38 GB) al disco del pod.
# Uso: bash scripts/setup_weights.sh [carpeta_destino]
# Por defecto /workspace/weights (que en RunPod suele ser el disco persistente).
#
# Tarda unos 5 minutos con buena conexion (aria2c baja en paralelo).
set -euo pipefail

W="${1:-/workspace/weights}"
WAN="$W/Wan2.1-I2V-14B-480P"
ITA="$W/InfiniteTalk"
QW="$ITA/quant_models"
W2V="$W/chinese-wav2vec2-base"

mkdir -p "$WAN" "$QW" "$W2V"
export HF_HOME="${HF_HOME:-/workspace/hf}"

BW="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main"
BI="https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main"

bajar() {  # url destino
  if [ -s "$2" ]; then echo "-- ya esta: $(basename "$2")"; return 0; fi
  echo "-- bajando $(basename "$2")"
  aria2c -q -c -x 8 -s 8 -k 1M -d "$(dirname "$2")" -o "$(basename "$2")" "$1"
}

echo "=== Pesos base de Wan 2.1 (VAE + T5 + CLIP)"
bajar "$BW/Wan2.1_VAE.pth" "$WAN/Wan2.1_VAE.pth"
bajar "$BW/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

echo "=== T5 comprimido (fp8, 6.7 GB) -- es el que evita el problema de memoria"
bajar "$BI/quant_models/t5_fp8.safetensors" "$QW/t5_fp8.safetensors"
bajar "$BI/quant_models/t5_map_fp8.json" "$QW/t5_map_fp8.json"

echo "=== Modelo de video InfiniteTalk comprimido (fp8, 19.5 GB)"
bajar "$BI/quant_models/infinitetalk_single_fp8.safetensors" "$QW/infinitetalk_single_fp8.safetensors"
bajar "$BI/quant_models/infinitetalk_single_fp8.json" "$QW/infinitetalk_single_fp8.json"
bajar "$BI/quant_models/quant.json" "$QW/quant.json"

echo "=== Configuracion y tokenizadores"
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN" --include "google/umt5-xxl/*" "config.json" "*.index.json"
hf download FacebookAI/xlm-roberta-large --local-dir "$WAN/xlm-roberta-large" --include "tokenizer*" "special_tokens_map.json" "sentencepiece.bpe.model" "config.json"

echo "=== Codificador de audio (chino-wav2vec2, revision pr/1)"
hf download TencentGameMate/chinese-wav2vec2-base --revision refs/pr/1 --local-dir "$W2V"

# El codigo busca el T5 en la carpeta padre de quant_models: dejamos el enlace.
ln -sf quant_models/t5_fp8.safetensors "$ITA/t5_fp8.safetensors"
ln -sf quant_models/t5_map_fp8.json "$ITA/t5_map_fp8.json"

echo
echo "=== Inventario (los tamanos tienen que decir GB, no 0)"
ls -la "$WAN"/*.pth "$WAN"/xlm-roberta-large/sentencepiece.bpe.model "$QW" "$W2V" 2>&1 | tail -25
du -sh "$W"/* 2>/dev/null
echo "LISTO: pesos en $W"
