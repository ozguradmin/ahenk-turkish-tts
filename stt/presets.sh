# shellcheck shell=bash
# Model presets shared by train.sh and export_and_test.sh (source this file).
# preset_args NAME  -> fills the bash arrays MODEL_ARGS and STREAM_ARGS and the
#                      variables PRESET_MAX_DURATION / PRESET_EPOCHS.
#
#   small  24.5 M params (vocab 1000; encoder 22.3 M): icefall "small" Zipformer2 (23.3 M with
#          vocab 500), causal. Target: phones / Raspberry Pi / WASM.
#   medium 67.1 M params (vocab 1000; encoder 64.6 M): icefall default Zipformer2, causal.
#   tiny   1.2 M params: CPU smoke test only (proves the pipeline, useless model).
#
# Streaming: trained with random chunk sizes 16/32/64 frames (at 50 Hz after the
# first subsampling = 320/640/1280 ms) and left contexts 64/128/256 frames, plus
# full context (-1). Export picks one, default chunk 16 (320 ms) + left 128.
#
# PRESET_MAX_DURATION is the seconds of audio per batch for an 8 GB GPU with
# fp16, derived from icefall's published training logs (README "VRAM"):
# peak allocated ~ 0.4 GB + 12.6 MB/s (small), ~ 1.3 GB + 15.7 MB/s (medium),
# incl. the vocab-1000 joiner. icefall's OOM scan at startup tries the worst
# batches first; if it fails, lower --max-duration by 50.

preset_args() {
  STREAM_ARGS=(--causal 1 --chunk-size 16,32,64,-1 --left-context-frames 64,128,256,-1)
  case "$1" in
    small)
      MODEL_ARGS=(
        --num-encoder-layers 2,2,2,2,2,2
        --downsampling-factor 1,2,4,8,4,2
        --feedforward-dim 512,768,768,768,768,768
        --num-heads 4,4,4,8,4,4
        --encoder-dim 192,256,256,256,256,256
        --encoder-unmasked-dim 192,192,192,192,192,192
        --query-head-dim 32 --value-head-dim 12 --pos-head-dim 4 --pos-dim 48
        --cnn-module-kernel 31,31,15,15,15,31
        --decoder-dim 512 --joiner-dim 512
      )
      PRESET_MAX_DURATION=400
      PRESET_EPOCHS=20
      ;;
    medium)
      MODEL_ARGS=(
        --num-encoder-layers 2,2,3,4,3,2
        --downsampling-factor 1,2,4,8,4,2
        --feedforward-dim 512,768,1024,1536,1024,768
        --num-heads 4,4,4,8,4,4
        --encoder-dim 192,256,384,512,384,256
        --encoder-unmasked-dim 192,192,256,256,256,192
        --query-head-dim 32 --value-head-dim 12 --pos-head-dim 4 --pos-dim 48
        --cnn-module-kernel 31,31,15,15,15,31
        --decoder-dim 512 --joiner-dim 512
      )
      PRESET_MAX_DURATION=250
      PRESET_EPOCHS=20
      ;;
    tiny)
      MODEL_ARGS=(
        --num-encoder-layers 1,1,1,1,1,1
        --downsampling-factor 1,2,4,8,4,2
        --feedforward-dim 128,128,128,128,128,128
        --num-heads 2,2,2,2,2,2
        --encoder-dim 64,64,64,64,64,64
        --encoder-unmasked-dim 64,64,64,64,64,64
        --query-head-dim 16 --value-head-dim 8 --pos-head-dim 4 --pos-dim 16
        --cnn-module-kernel 15,15,15,15,15,15
        --decoder-dim 64 --joiner-dim 64
      )
      PRESET_MAX_DURATION=60
      PRESET_EPOCHS=1
      ;;
    *)
      echo "unknown preset '$1' (small | medium | tiny)" >&2
      return 1
      ;;
  esac
}
