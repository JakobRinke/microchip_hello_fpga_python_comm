"""Export a row-aligned Keras CNN to sparse DDR `.mem` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convert_to_hex_dense import (
    SCRIPT_DIR,
    build_stages,
    control_word,
    export_fc_words,
    export_metadata,
    keras_pipeline,
    load_model_compatible,
    quantize_bias,
    quantize_signed16,
    resolve_model_path,
    validate_ddr_layout,
    write_mem32,
)


PARALLEL_CONV = 8
KERNEL_SIZE = 3
RECORD_BITS = 1024
RECORD_WORDS_32 = 32
WEIGHT_BITS = 18
BIAS_OFFSET = 432
CONTROL_OFFSET = 560


def customize_quantized_sparse_weights(
    weights: np.ndarray, layer_index: int, layer_name: str
) -> np.ndarray:
    """Optional student hook, applied after INT16 quantization and before packing.

    ``weights`` uses Keras order [kernel_row, kernel_column, input_channel,
    output_channel]. Row-aligned hardware permits at most one nonzero among the
    three columns for every row/input/output combination.
    """
    del layer_index, layer_name  # Available for layer-specific conditions.

    # Examples (uncomment only what you need):
    # weights[:] = 0                         # every sparse Conv weight = 0
    # weights[weights != 0] = 1              # retained weights = 1; keep positions
    # weights[:] = 0                         # choose new row-aligned positions
    # weights[:, 1, :, :] = 1                # one value at column 1 in every row
    # weights[0, :, 0, 0] = 0
    # weights[0, 2, 0, 0] = 123              # custom value/column for one row

    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sparse 1024-bit DDR accelerator records.")
    parser.add_argument("model", nargs="?", type=Path, default=Path("mnist_cnn_32_64_sparse.h5"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "sparse_generated",
        help="generated output directory (default: main_files/sparse_generated)",
    )
    parser.add_argument("--prefix", default=None, help="optional output filename prefix")
    parser.add_argument("--frac-bits", type=int, default=14, help="fixed-point fractional bits")
    parser.add_argument(
        "--keep-conv-biases",
        action="store_true",
        help="export quantized conv biases; default keeps reference behavior and writes zeros",
    )
    return parser.parse_args()


def encode_sparse_row(row: np.ndarray) -> int:
    """Encode one row exactly like the original sparse exporter."""

    nonzero = np.flatnonzero(row)
    if len(nonzero) > 1:
        raise ValueError(f"sparse kernel row contains {len(nonzero)} nonzeros, expected at most one")
    if len(nonzero) == 0:
        return 0
    column = int(nonzero[0])
    value = int(row[column])
    # Match the RTL column order.
    return ((2 - column) << 16) | (value & 0xFFFF)


def set_bits(record: int, offset: int, width: int, value: int) -> int:
    mask = (1 << width) - 1
    return record | ((int(value) & mask) << offset)


def make_sparse_record(kernels: np.ndarray, biases: np.ndarray, control: int) -> int:
    if kernels.shape != (PARALLEL_CONV, KERNEL_SIZE, KERNEL_SIZE):
        raise ValueError(f"expected kernels shape (8, 3, 3), got {kernels.shape}")
    if biases.shape != (PARALLEL_CONV,):
        raise ValueError(f"expected biases shape (8,), got {biases.shape}")

    record = 0
    encoded_rows: list[int] = []
    for kernel in kernels:
        for row in kernel[:, ::-1]:
            encoded_rows.append(encode_sparse_row(row))
    for index, value in enumerate(encoded_rows):
        record = set_bits(record, index * WEIGHT_BITS, WEIGHT_BITS, value)
    for index, value in enumerate(biases):
        record = set_bits(record, BIAS_OFFSET + index * 16, 16, int(value))
    record = set_bits(record, CONTROL_OFFSET, 16, control)
    return record


def _decode_signed16(value: int) -> int:
    return int(np.array([value & 0xFFFF], dtype=np.uint16).view(np.int16)[0])


def decode_sparse_record(record: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode one sparse record back to original-width kernel rows."""
    kernels = np.zeros((PARALLEL_CONV, KERNEL_SIZE, KERNEL_SIZE), dtype=np.int16)
    for lane in range(PARALLEL_CONV):
        for row in range(KERNEL_SIZE):
            index = lane * KERNEL_SIZE + row
            encoded = (record >> (index * WEIGHT_BITS)) & ((1 << WEIGHT_BITS) - 1)
            value = _decode_signed16(encoded)
            column = (encoded >> 16) & 0x3
            if value:
                if column >= KERNEL_SIZE:
                    raise ValueError(f"decoded invalid sparse column {column}")
                kernels[lane, row, column] = value

    biases = np.array(
        [
            _decode_signed16(record >> (BIAS_OFFSET + index * 16))
            for index in range(PARALLEL_CONV)
        ],
        dtype=np.int16,
    )
    control = (record >> CONTROL_OFFSET) & 0xFFFF
    return kernels, biases, control


def validate_sparse_record_roundtrip(
    conv_pipeline,
    stages,
    records: list[int],
    scale: int,
    keep_conv_biases: bool,
) -> int:
    """Verify every packed sparse record decodes to its expected Q2.14 tensors."""
    record_index = 0
    for layer_index, ((layer, _), stage) in enumerate(zip(conv_pipeline, stages)):
        raw_weights, raw_biases = layer.get_weights()
        weights = quantize_signed16(raw_weights, scale)
        weights = customize_quantized_sparse_weights(weights, layer_index, layer.name)
        biases = quantize_bias(raw_biases, scale)
        if not keep_conv_biases:
            biases = np.zeros_like(biases)

        for output_tile in range(stage.output_tiles):
            for offset in range(stage.offset_count):
                for input_tile in range(stage.input_tiles):
                    input_channel = 0 if layer_index == 0 else (input_tile * PARALLEL_CONV) + offset
                    expected_kernels = np.zeros(
                        (PARALLEL_CONV, KERNEL_SIZE, KERNEL_SIZE), dtype=np.int16
                    )
                    expected_biases = np.zeros(PARALLEL_CONV, dtype=np.int16)
                    for lane in range(PARALLEL_CONV):
                        output_channel = output_tile * PARALLEL_CONV + lane
                        expected_kernels[lane] = weights[:, :, input_channel, output_channel]
                        expected_biases[lane] = biases[output_channel]
                    pool_now = stage.use_pool and input_tile == stage.input_tiles - 1
                    expected_control = control_word(
                        ram_dir=1 if layer_index == 0 else output_tile & 1,
                        img_size=stage.input_size,
                        use_pool=pool_now,
                        in_layer_size=int(np.log2(stage.offset_count)),
                        write_offset=offset,
                    )

                    decoded_kernels, decoded_biases, decoded_control = decode_sparse_record(
                        records[record_index]
                    )
                    if not np.array_equal(decoded_kernels, expected_kernels):
                        raise ValueError(f"sparse record {record_index}: kernel round-trip mismatch")
                    if not np.array_equal(decoded_biases, expected_biases):
                        raise ValueError(f"sparse record {record_index}: bias round-trip mismatch")
                    if decoded_control != expected_control:
                        raise ValueError(f"sparse record {record_index}: control round-trip mismatch")
                    record_index += 1

    if record_index != len(records):
        raise ValueError(f"validated {record_index} sparse records, but exporter produced {len(records)}")
    return record_index


def record_words(record: int) -> list[int]:
    return [(record >> (32 * index)) & 0xFFFF_FFFF for index in range(RECORD_WORDS_32)]


def export_conv_records(conv_pipeline, stages, scale: int, keep_conv_biases: bool) -> list[int]:
    records: list[int] = []
    for layer_index, ((layer, _), stage) in enumerate(zip(conv_pipeline, stages)):
        raw_weights, raw_biases = layer.get_weights()
        weights = quantize_signed16(raw_weights, scale)
        weights = customize_quantized_sparse_weights(weights, layer_index, layer.name)
        biases = quantize_bias(raw_biases, scale)
        if not keep_conv_biases:
            biases = np.zeros_like(biases)

        for output_tile in range(stage.output_tiles):
            for offset in range(stage.offset_count):
                for input_tile in range(stage.input_tiles):
                    input_channel = 0 if layer_index == 0 else (input_tile * PARALLEL_CONV) + offset
                    first_output = output_tile * PARALLEL_CONV
                    output_slice = slice(first_output, first_output + PARALLEL_CONV)
                    kernels = weights[:, :, input_channel, output_slice].transpose(2, 0, 1)
                    record_biases = biases[output_slice]
                    pool_now = stage.use_pool and input_tile == stage.input_tiles - 1
                    control = control_word(
                        ram_dir=1 if layer_index == 0 else output_tile & 1,
                        img_size=stage.input_size,
                        use_pool=pool_now,
                        in_layer_size=int(np.log2(stage.offset_count)),
                        write_offset=offset,
                    )
                    records.append(make_sparse_record(kernels, record_biases, control))
    return records


def write_tb_parameters(
    path: Path,
    *,
    stages,
    records: list[int],
    fc_pair_count: int,
    class_count: int,
    metadata_path: Path,
    conv_path: Path,
    fc_path: Path,
) -> None:
    conv_word_count = len(records) * RECORD_WORDS_32
    fc_word_count = fc_pair_count * class_count
    expected_replay_writes = sum(
        stage.output_tiles * stage.input_tiles * (stages[index - 1].output_word_count // 4)
        for index, stage in enumerate(stages)
        if index > 0
    )
    lines = [
        "Sparse TOP_BIG_CNN_TB model parameters",
        "======================================",
        "",
        "Generated by python/main_files/sparse_hex_file.py.",
        "Each convolution record is 1024 bits, or 32 DDR words.",
        "",
        "Model files",
        "-----------",
        f'BIG_MODEL_METADATA_FILE     = "{metadata_path.name}"',
        f'BIG_MODEL_CONV_RECORDS_FILE = "{conv_path.name}"',
        f'BIG_MODEL_FC_WEIGHTS_FILE   = "{fc_path.name}"',
        "",
        "Architecture",
        "------------",
    ]
    for index, stage in enumerate(stages, start=1):
        pool_text = " -> pool2x2" if stage.use_pool else ""
        lines.append(
            f"Conv {index}: {stage.input_channels} -> {stage.output_channels} channels, "
            f"{stage.input_size}x{stage.input_size} -> {stage.output_size}x{stage.output_size}"
            f"{pool_text}"
        )
    lines.extend(
        [
            f"FC: {fc_pair_count * 2} inputs -> {class_count} classes",
            "",
            "Testbench constants",
            "-------------------",
            "DDR_BASE_ADDR          = 32'hA0000000",
            "CONV_RECORD_BASE_ADDR  = 32'hA0001000",
            "FC_WEIGHT_BASE_ADDR    = 32'hA0012000",
            f"META_WORDS             = {8 + 4 * 8}",
            f"RECORD_WORDS           = {RECORD_WORDS_32}",
            f"EXPECTED_CONV_RECORDS  = {len(records)}",
            f"CONV_RECORD_WORD_COUNT = {conv_word_count}",
            f"FC_INPUT_PAIR_COUNT    = {fc_pair_count}",
            f"FC_CLASS_COUNT         = {class_count}",
            f"FC_WEIGHT_WORD_COUNT   = {fc_word_count}",
            f"EXPECTED_REPLAY_WRITES = {expected_replay_writes}",
            "",
            "Per-layer DDR map",
            "-----------------",
        ]
    )
    for index, stage in enumerate(stages, start=1):
        lines.extend(
            [
                f"Conv {index}:",
                f"  INPUT_TILES          = {stage.input_tiles}",
                f"  OUTPUT_TILES         = {stage.output_tiles}",
                f"  OFFSET_COUNT         = {stage.offset_count}",
                f"  INPUT_TILE_WORDS     = {stage.input_word_count}",
                f"  OUTPUT_TILE_WORDS    = {stage.output_word_count}",
                f"  RECORD_BASE_INDEX    = {stage.record_base}",
                f"  RECORD_COUNT         = {stage.record_count}",
                f"  INPUT_BASE_ADDR      = 32'h{stage.input_base_addr:08X}",
                f"  OUTPUT_BASE_ADDR     = 32'h{stage.output_base_addr:08X}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    if not 0 <= args.frac_bits <= 15:
        raise ValueError("--frac-bits must be between 0 and 15")
    scale = 1 << args.frac_bits
    model_path = resolve_model_path(args.model)
    output_dir = args.output_dir if args.output_dir.is_absolute() else SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_compatible(model_path)
    conv_pipeline, fc_layer = keras_pipeline(model)
    stages = build_stages(conv_pipeline)
    records = export_conv_records(conv_pipeline, stages, scale, args.keep_conv_biases)
    fc_words, fc_pair_count, class_count = export_fc_words(fc_layer, stages[-1], scale)
    metadata = export_metadata(stages, fc_pair_count, class_count)

    channel_suffix = "_".join(str(stage.output_channels) for stage in stages)
    prefix = args.prefix or f"model_conv{channel_suffix}"
    conv_path = output_dir / f"{prefix}_records_sparse.mem"
    metadata_path = output_dir / f"metadata_{prefix}_sparse.mem"
    fc_path = output_dir / f"{prefix}_fc_weights_sparse.mem"

    conv_words = [word for record in records for word in record_words(record)]
    conv_storage_bytes = len(conv_words) * 4
    metadata_storage_bytes = len(metadata) * 4
    fc_storage_bytes = len(fc_words) * 4
    model_storage_bytes = conv_storage_bytes + metadata_storage_bytes + fc_storage_bytes
    validate_ddr_layout(stages, len(conv_words), len(fc_words), fc_pair_count, class_count)
    write_mem32(conv_path, conv_words)
    write_mem32(metadata_path, metadata)
    write_mem32(fc_path, fc_words)

    print(f"Loaded model:       {model_path}")
    for index, stage in enumerate(stages, start=1):
        print(
            f"Conv {index}:            {stage.input_channels:>3} -> {stage.output_channels:<3} "
            f"input={stage.input_size:>2} output={stage.output_size:>2} "
            f"pool={stage.use_pool} records={stage.record_count}"
        )
    print(f"Total conv records: {len(records)}")
    print(f"Conv DDR words:     {len(conv_words)}")
    print(f"FC weight words:    {len(fc_words)}")
    print(f"Model DDR bytes:    {model_storage_bytes}")
    print(f"Metadata:           {metadata_path}")
    print(f"Conv records:       {conv_path}")
    print(f"FC weights:         {fc_path}")


if __name__ == "__main__":
    main()

# CUSTOM INT16 ROW-ALIGNED WEIGHTS
# Put these lines inside customize_quantized_sparse_weights() above.
# Valid values: -32768 to 32767; keep at most one nonzero per 3-value row.
# weights[:] = 0
# weights[:, 1, :, :] = np.int16(123)        # column 1 in every row
# weights[0, :, 0, 0] = 0
# weights[0, 2, 0, 0] = np.int16(-456)       # one row uses column 2
# weights[weights != 0] = np.int16(1000)     # change retained values only
