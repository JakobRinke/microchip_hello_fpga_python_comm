"""Single-conv-layer network: config memory, zero FC memory, constant conv records (dense)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import convert_to_hex_dense as cvt

# ---- Netzwerk-Konfiguration ------------------------------------------------
OUT_CHANNELS = 128      # 8 oder 16 (Vielfaches von 8)
NUM_CLASSES = 1
USE_POOL = True
FRAC_BITS = 14

OUTPUT_DIR = Path("generated_single_conv")

def to_32bit_hex(num: int) -> str:
    """Hilfsfunktion: int32 in hex-String umwandeln (für .mem-Dateien)."""
    return f"0x{num & 0xFFFFFFFF:08X}"

def make_8column_comma_array(arr: np.ndarray) -> str:
    """Hilfsfunktion: 1D-Array in 8-Spalten-Text umwandeln (für .mem-Dateien)."""
    lines = []
    for i in range(0, len(arr), 8):
        line = ", ".join(f"{to_32bit_hex(x)}U" for x in arr[i:i + 8]) + ","
        lines.append(line)
    return "\n".join(lines)


def build_network(
    out_channels: int = OUT_CHANNELS,
    num_classes: int = NUM_CLASSES,
    use_pool: bool = USE_POOL,
):
    """Baut conv_pipeline, fc_layer und stages ohne Modelldatei (alle Gewichte 0)."""
    conv = cvt.H5Layer(
        class_name="Conv2D",
        name="conv2d_0",
        weights=[
            np.zeros((3, 3, 1, out_channels), dtype=np.float32),
            np.zeros(out_channels, dtype=np.float32),
        ],
        config={
            "strides": (1, 1),
            "padding": "valid",
            "dilation_rate": (1, 1),
            "groups": 1,
        },
    )
    conv_pipeline = [(conv, use_pool)]
    stages = cvt.build_stages(conv_pipeline)

    final = stages[-1]
    n_features = final.output_channels * final.output_size * final.output_size
    fc_layer = cvt.H5Layer(
        class_name="Dense",
        name="dense_0",
        weights=[
            np.zeros((n_features, num_classes), dtype=np.float32),
            np.zeros(num_classes, dtype=np.float32),
        ],
        units=num_classes,
    )
    return conv_pipeline, fc_layer, stages


def get_config_memory(**net_kwargs) -> np.ndarray:
    """Metadata/Konfigurations-Memory als uint32-Array."""
    conv_pipeline, fc_layer, stages = build_network(**net_kwargs)
    scale = 1 << FRAC_BITS
    _, fc_pair_count, class_count = cvt.export_fc_words(fc_layer, stages[-1], scale)
    metadata = cvt.export_metadata(stages, fc_pair_count, class_count)
    return np.array(metadata, dtype=np.uint32)


def get_fc_zero_memory(**net_kwargs) -> np.ndarray:
    """FC-Weights-Memory (alle 0) als uint32-Array."""
    conv_pipeline, fc_layer, stages = build_network(**net_kwargs)
    scale = 1 << FRAC_BITS
    fc_words, _, _ = cvt.export_fc_words(fc_layer, stages[-1], scale)
    return np.array(fc_words, dtype=np.uint32)


def constant_weights_dense(weights, layer_index, layer_name, const_value):
    weights[:] = np.int16(const_value)
    return weights


def create_constant_weight_stream_dense(const_value: int = 256, **net_kwargs) -> np.ndarray:
    """Conv-Records (uint32-Wörter), alle Conv-Weights = const_value."""
    if not -32768 <= int(const_value) <= 32767:
        raise ValueError("const_value muss in int16 passen (-32768..32767)")

    original_hook = cvt.customize_quantized_conv_weights
    cvt.customize_quantized_conv_weights = (
        lambda weights, layer_index, layer_name: constant_weights_dense(
            weights, layer_index, layer_name, const_value
        )
    )
    try:
        conv_pipeline, _, stages = build_network(**net_kwargs)
        scale = 1 << FRAC_BITS
        records = cvt.export_conv_records(
            conv_pipeline, stages, scale, keep_conv_biases=False
        )
    finally:
        cvt.customize_quantized_conv_weights = original_hook  # Hook wieder zurücksetzen

    conv_words = cvt.pack_signed16_pairs(np.asarray(records, dtype=np.int16).reshape(-1))
    return np.array(conv_words, dtype=np.uint32)


def main() -> None:
    conv_pipeline, fc_layer, stages = build_network()
    config_mem = get_config_memory()
    fc_mem = get_fc_zero_memory()
    conv_mem = create_constant_weight_stream_dense(const_value=256)

    # gleiche DDR-Layout-Prüfung wie im Original-Exporter
    cvt.validate_ddr_layout(
        stages, len(conv_mem), len(fc_mem), len(fc_mem) // NUM_CLASSES, NUM_CLASSES
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"single_conv{OUT_CHANNELS}"
    cvt.write_mem32(OUTPUT_DIR / f"metadata_{prefix}_dense.mem", config_mem.tolist())
    cvt.write_mem32(OUTPUT_DIR / f"{prefix}_fc_weights_dense.mem", fc_mem.tolist())
    cvt.write_mem32(OUTPUT_DIR / f"{prefix}_records_dense.mem", conv_mem.tolist())

    with open(OUTPUT_DIR / f"{prefix}_records_dense.c", "w") as f:
        f.write(make_8column_comma_array(conv_mem))
    with open(OUTPUT_DIR / f"{prefix}_fc_weights_dense.c", "w") as f:
        f.write(make_8column_comma_array(fc_mem))
    with open(OUTPUT_DIR / f"metadata_{prefix}_dense.c", "w") as f:
        f.write(make_8column_comma_array(config_mem))


    print(f"Config memory:   {config_mem.shape} words")
    print(f"FC zero memory:  {fc_mem.shape} words")
    print(f"Conv records:    {conv_mem.shape} words ({len(conv_mem) // cvt.RECORD_WORDS_32} records)")


if __name__ == "__main__":
    main()