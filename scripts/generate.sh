#!/bin/bash
# Genera un video hablando a partir de una foto y un audio.
#
# Uso: bash scripts/generate.sh FOTO AUDIO SALIDA [PASOS] [PROMPT] [CUADROS]
#   FOTO    imagen de la cara (jpg/png)
#   AUDIO   wav/mp3 con la voz (define el largo del video: 5 s de audio = 5 s de video)
#   SALIDA  archivo mp4 de salida
#   PASOS   pasos de muestreo (20 por defecto; mas pasos = mejor y mas lento)
#   CUADROS cantidad de cuadros (4n+1). El video dura CUADROS/25 segundos, NO lo que
#           dure el audio: 81 = 3,2 s · 125 = 5 s · 249 = 10 s. El costo sube con esto.
#
# Variables opcionales:
#   WEIGHTS=/workspace/weights      carpeta con los modelos
#   DIT_PARAMS=0                    parametros que quedan residentes en la placa.
#                                   Con 16 GB dejar 0 (mas lento pero entra).
#                                   Con 40 GB o mas, subirlo acelera mucho.
#   TAMANO=infinitetalk-480         o infinitetalk-720 (mas pesado)
set -euo pipefail

# Por SSH la sesion NO hereda el ENV de la imagen: sin esto, python3 es el del
# sistema y no encuentra diffusers/transformers del venv.
export PATH=/opt/venv/bin:$PATH

FOTO="${1:?falta la foto}"
AUDIO="${2:?falta el audio}"
SALIDA="${3:-/workspace/salida.mp4}"
PASOS="${4:-20}"
PROMPT="${5:-A person talking to the camera, natural mouth movement, cinematic lighting, high detail}"
CUADROS="${6:-81}"

W="${WEIGHTS:-/workspace/weights}"
Q="$W/InfiniteTalk/quant_models"
DIT_PARAMS="${DIT_PARAMS:-0}"
TAMANO="${TAMANO:-infinitetalk-480}"

for f in "$W/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
         "$Q/infinitetalk_single_fp8.safetensors" "$Q/t5_fp8.safetensors" \
         "$W/chinese-wav2vec2-base"; do
  [ -e "$f" ] || { echo "FALTA $f -- corre primero scripts/setup_weights.sh"; exit 1; }
done
[ -f "$FOTO" ] || { echo "no encuentro la foto: $FOTO"; exit 1; }
[ -f "$AUDIO" ] || { echo "no encuentro el audio: $AUDIO"; exit 1; }

IN=/tmp/in.json
python3 - "$PROMPT" "$FOTO" "$AUDIO" "$IN" <<'PY'
import json, sys
prompt, foto, audio, salida = sys.argv[1:5]
json.dump({"prompt": prompt, "cond_video": foto, "cond_audio": {"person1": audio}},
          open(salida, "w"), ensure_ascii=False, indent=2)
print("in.json listo")
PY

cd /opt/italk
echo "=== generando ($PASOS pasos, $CUADROS cuadros = $(( CUADROS / 25 )) s aprox, $TAMANO, dit_params=$DIT_PARAMS)"
# OJO con --quant_dir: el codigo espera el ARCHIVO .safetensors, no la carpeta.
# OJO con --t5_cpu: con ese flag el texto se codifica mal y el modelo recibe una lista
# en vez de un tensor (AttributeError: 'list' object has no attribute 'dtype').
python3 -u generate_infinitetalk.py \
  --ckpt_dir "$W/Wan2.1-I2V-14B-480P" \
  --infinitetalk_dir "$Q/infinitetalk_single_fp8.safetensors" \
  --quant_dir "$Q/infinitetalk_single_fp8.safetensors" \
  --quant fp8 \
  --wav2vec_dir "$W/chinese-wav2vec2-base" \
  --input_json "$IN" \
  --size "$TAMANO" \
  --frame_num "$CUADROS" \
  --sample_steps "$PASOS" \
  --motion_frame 9 \
  --num_persistent_param_in_dit "$DIT_PARAMS" \
  --mode clip --use_teacache \
  --save_file "$SALIDA"

ls -la "$SALIDA"
echo "LISTO: $SALIDA"
