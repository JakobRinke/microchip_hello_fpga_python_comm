"""Export a Keras CNN to dense DDR `.mem` files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent

PARALLEL_CONV = 8
KERNEL_SIZE = 3
RECORD_WORDS_16 = 128
RECORD_WORDS_32 = 64
MAX_CONV_LAYERS = 4
MAX_CONV_RECORDS = 512
SUPPORTED_OFFSET_COUNTS = {1, 2, 4, 8}

MODEL_METADATA_BASE_ADDR = 0xA000_0000
CONV_RECORD_BASE_ADDR = 0xA000_1000
FC_WEIGHT_BASE_ADDR = 0xA001_2000
ACTIVATION_BASE_ADDR = 0xA002_0000
ACTIVATION_REGION_BYTES = 0x0001_0000


@dataclass
class ConvStage:
    name: str
    input_channels: int
    output_channels: int
    input_size: int
    output_size: int
    use_pool: bool
    input_tiles: int
    output_tiles: int
    offset_count: int
    input_word_count: int
    output_word_count: int
    record_base: int
    record_count: int
    input_base_addr: int
    output_base_addr: int


@dataclass
class H5Layer:
    """Small compatibility wrapper for Keras 3 HDF5 files read by TensorFlow 2."""

    class_name: str
    name: str
    weights: list[np.ndarray]
    units: int | None = None
    config: dict[str, object] | None = None

    def get_weights(self) -> list[np.ndarray]:
        return self.weights


@dataclass
class H5Model:
    layers: list[H5Layer]
    input_shape: tuple[object, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained Keras CNN into dense 2048-bit DDR accelerator records."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("mnist_cnn_32_64.h5"),
        help="input Keras .h5 model (default: mnist_cnn_32_64.h5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "generated_dense",
        help="directory for generated files (default: main_files/generated_dense)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="output filename prefix; default is derived from conv channels",
    )
    parser.add_argument(
        "--frac-bits",
        type=int,
        default=14,
        help="signed fixed-point fractional bits (default: 14)",
    )
    parser.add_argument(
        "--keep-conv-biases",
        action="store_true",
        help=(
            "export quantized Conv2D biases; by default biases are zeroed to preserve "
            "the reference exporter behavior"
        ),
    )
    parser.add_argument(
        "--fc-weight-base-addr",
        type=lambda value: int(value, 0),
        default=None,
        help=(
            "override the DDR base address used for FC weights, e.g. 0xA0016000; "
            "useful when a deeper dense model needs more convolution-record space"
        ),
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path) -> Path:
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"model does not exist: {model_path}")
    return model_path


def read_h5_dataset(group: h5py.Group, basename: str) -> np.ndarray:
    matches: list[np.ndarray] = []

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset) and name.split("/")[-1].split(":")[0] == basename:
            matches.append(np.asarray(item))

    group.visititems(collect)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {basename!r} dataset below {group.name}, found {len(matches)}"
        )
    return matches[0]


def load_h5_model(model_path: Path) -> H5Model:
    """Read layer configuration and tensors without relying on Keras deserialization."""

    layers: list[H5Layer] = []
    input_shape: tuple[object, ...] | None = None
    with h5py.File(model_path, "r") as h5_file:
        raw_config = h5_file.attrs["model_config"]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        config = json.loads(raw_config)
        model_weights = h5_file["model_weights"]

        for layer_config in config["config"]["layers"]:
            class_name = layer_config["class_name"]
            details = layer_config["config"]
            name = details["name"]
            if class_name == "InputLayer":
                raw_shape = details.get("batch_shape", details.get("batch_input_shape"))
                if raw_shape:
                    input_shape = tuple(raw_shape[1:])
            weights: list[np.ndarray] = []
            if class_name in {"Conv2D", "Dense"}:
                group = model_weights[name]
                weights = [read_h5_dataset(group, "kernel"), read_h5_dataset(group, "bias")]
            layers.append(
                H5Layer(
                    class_name=class_name,
                    name=name,
                    weights=weights,
                    units=details.get("units"),
                    config=details,
                )
            )
    return H5Model(layers, input_shape)


def load_model_compatible(model_path: Path) -> tf.keras.Model | H5Model:
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as exc:
        print(f"TensorFlow loader could not deserialize this .h5 file: {exc}")
        print("Using the HDF5 compatibility reader; tensor values are unchanged.")
        return load_h5_model(model_path)


def quantize_signed16(values: np.ndarray, scale: int) -> np.ndarray:
    quantized = np.round(values * scale)
    return np.clip(quantized, -32768, 32767).astype(np.int16)


def customize_quantized_conv_weights(
    weights: np.ndarray, layer_index: int, layer_name: str
) -> np.ndarray:
    """Optional student hook, applied after INT16 quantization and before packing.

    ``weights`` uses Keras order [kernel_row, kernel_column, input_channel,
    output_channel]. Leave this function unchanged to export the model weights.
    To create a test image, uncomment or replace the assignments below.
    """
    del layer_index, layer_name  # Available for layer-specific conditions.

    # Examples (uncomment only what you need):
    # weights[:] = 0                         # every dense Conv weight = 0
    # weights[:] = 1                         # every dense Conv weight = 1
    # weights[0, 0, 0, 0] = 123             # change one quantized weight
    # weights[:, :, :, 0] = -7               # output-channel 0 only

    return weights


def quantize_bias(values: np.ndarray, scale: int) -> np.ndarray:
    quantized = np.round(values * scale)
    return np.clip(quantized, -128, 127).astype(np.int16)


def as_uint16(value: int) -> int:
    return int(value) & 0xFFFF


def as_uint32(value: int) -> int:
    return int(value) & 0xFFFF_FFFF


def pack_signed16_pairs(values: list[int] | np.ndarray) -> list[int]:
    if len(values) % 2:
        raise ValueError("16-bit value count must be even before packing into 32-bit words")
    packed = np.asarray(values, dtype=np.int16).view(np.uint16).astype(np.uint32)
    return (packed[0::2] | (packed[1::2] << 16)).tolist()


def write_mem32(path: Path, words: list[int]) -> None:
    path.write_text("".join(f"{as_uint32(word):08X}\n" for word in words), encoding="ascii")


def control_word(
    *,
    ram_dir: int,
    img_size: int,
    use_pool: bool,
    in_layer_size: int,
    write_offset: int,
) -> int:
    value = 0
    value |= (ram_dir & 0x1) << 0
    value |= (img_size & 0x1F) << 1
    value |= (int(use_pool) & 0x1) << 6
    value |= (in_layer_size & 0x3) << 7
    value |= (write_offset & 0x7) << 9
    return value


def make_record(kernels: np.ndarray, biases: np.ndarray, control: int) -> list[int]:
    """Create one reference-compatible dense 2048-bit record.

    ``kernels`` has shape ``(8, 3, 3)``. Reversing the kernel width preserves
    the original exporter's Verilog ordering.
    """

    if kernels.shape != (PARALLEL_CONV, KERNEL_SIZE, KERNEL_SIZE):
        raise ValueError(f"expected kernels with shape (8, 3, 3), got {kernels.shape}")
    if biases.shape != (PARALLEL_CONV,):
        raise ValueError(f"expected biases with shape (8,), got {biases.shape}")

    values = [0] * RECORD_WORDS_16
    flattened_weights = kernels[:, :, ::-1].reshape(-1)
    values[:72] = [int(value) for value in flattened_weights]
    values[72:80] = [int(value) for value in biases]
    values[80] = int(control)
    return values


def _layer_setting(layer: object, name: str, default: object = None) -> object:
    if isinstance(layer, H5Layer):
        return (layer.config or {}).get(name, default)
    return getattr(layer, name, default)


def _pair(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value, value
    pair = tuple(int(item) for item in value)
    return pair if len(pair) == 2 else None


def keras_pipeline(model: tf.keras.Model | H5Model) -> tuple[list[tuple[object, bool]], object]:
    conv_pipeline: list[tuple[object, bool]] = []
    dense_layers: list[object] = []

    input_shape = getattr(model, "input_shape", None)
    if input_shape is not None:
        shape = tuple(input_shape)
        if len(shape) == 4:
            shape = shape[1:]
        if shape != (28, 28, 1):
            raise ValueError(f"hardware expects input shape (28, 28, 1), got {shape}")

    for layer in model.layers:
        class_name = getattr(layer, "class_name", layer.__class__.__name__)
        if class_name == "Conv2D":
            if _pair(_layer_setting(layer, "strides", (1, 1))) != (1, 1):
                raise ValueError(f"{layer.name}: hardware supports only stride-1 convolutions")
            if str(_layer_setting(layer, "padding", "valid")).lower() != "valid":
                raise ValueError(f"{layer.name}: hardware supports only valid convolutions")
            if _pair(_layer_setting(layer, "dilation_rate", (1, 1))) != (1, 1):
                raise ValueError(f"{layer.name}: hardware does not support dilated convolutions")
            if int(_layer_setting(layer, "groups", 1)) != 1:
                raise ValueError(f"{layer.name}: hardware does not support grouped convolutions")
            conv_pipeline.append((layer, False))
        elif class_name == "MaxPooling2D":
            if not conv_pipeline:
                raise ValueError("pooling appears before the first convolution")
            conv_layer, already_pooled = conv_pipeline[-1]
            if already_pooled:
                raise ValueError(f"{layer.name}: multiple pooling layers after one convolution")
            pool_size = _pair(_layer_setting(layer, "pool_size", (2, 2)))
            strides = _pair(_layer_setting(layer, "strides", pool_size))
            padding = str(_layer_setting(layer, "padding", "valid")).lower()
            if pool_size != (2, 2) or strides != (2, 2) or padding != "valid":
                raise ValueError(f"{layer.name}: hardware supports only valid 2x2 stride-2 pooling")
            conv_pipeline[-1] = (conv_layer, True)
        elif class_name == "Dense":
            dense_layers.append(layer)

    if not conv_pipeline:
        raise ValueError("model has no Conv2D layers")
    if len(conv_pipeline) > MAX_CONV_LAYERS:
        raise ValueError(f"hardware supports at most {MAX_CONV_LAYERS} convolution layers")
    if len(dense_layers) != 1:
        raise ValueError("hardware workflow expects exactly one final Dense layer")
    return conv_pipeline, dense_layers[0]


def build_stages(
    conv_pipeline: list[tuple[object, bool]],
) -> list[ConvStage]:
    stages: list[ConvStage] = []
    current_size = 28
    previous_output_words = 0
    next_record_base = 0

    for layer_index, (layer, use_pool) in enumerate(conv_pipeline):
        weights, _ = layer.get_weights()
        kernel_h, kernel_w, input_channels, output_channels = weights.shape
        if (kernel_h, kernel_w) != (KERNEL_SIZE, KERNEL_SIZE):
            raise ValueError(f"{layer.name}: hardware supports only 3x3 kernels")
        if layer_index == 0 and input_channels != 1:
            raise ValueError(f"{layer.name}: first convolution must have exactly one input channel")
        if layer_index and input_channels != stages[-1].output_channels:
            raise ValueError(
                f"{layer.name}: input channels {input_channels} do not match the previous "
                f"layer's {stages[-1].output_channels} output channels"
            )
        if output_channels % PARALLEL_CONV:
            raise ValueError(f"{layer.name}: output channels must be a multiple of 8")
        if layer_index and input_channels % PARALLEL_CONV:
            raise ValueError(f"{layer.name}: non-first-layer input channels must be a multiple of 8")
        if current_size > 0x1F:
            raise ValueError(f"{layer.name}: input size {current_size} does not fit the control record")

        conv_output_size = current_size - KERNEL_SIZE + 1
        if conv_output_size <= 0:
            raise ValueError(f"{layer.name}: spatial dimensions became invalid")
        output_size = conv_output_size // 2 if use_pool else conv_output_size
        if output_size <= 0:
            raise ValueError(f"{layer.name}: spatial dimensions became invalid after pooling")

        input_tiles = 1 if layer_index == 0 else input_channels // PARALLEL_CONV
        offset_count = 1 if layer_index == 0 else PARALLEL_CONV
        if offset_count not in SUPPORTED_OFFSET_COUNTS:
            raise ValueError(f"{layer.name}: unsupported offset count {offset_count}")
        output_tiles = output_channels // PARALLEL_CONV
        record_count = output_tiles * input_tiles * offset_count
        output_words = output_size * output_size * (PARALLEL_CONV // 2)
        if input_tiles > 0xFF or output_tiles > 0xFF or offset_count > 0xFF:
            raise ValueError(f"{layer.name}: tile counts do not fit scheduler metadata fields")
        if previous_output_words > 0xFFFF or output_words > 0xFFFF:
            raise ValueError(f"{layer.name}: activation word count does not fit scheduler metadata")
        activation_bytes = output_tiles * output_words * 4
        if activation_bytes > ACTIVATION_REGION_BYTES:
            raise ValueError(
                f"{layer.name}: output activations need {activation_bytes} bytes, but the "
                f"fixed DDR activation region allows {ACTIVATION_REGION_BYTES}"
            )

        stage = ConvStage(
            name=layer.name,
            input_channels=input_channels,
            output_channels=output_channels,
            input_size=current_size,
            output_size=output_size,
            use_pool=use_pool,
            input_tiles=input_tiles,
            output_tiles=output_tiles,
            offset_count=offset_count,
            input_word_count=previous_output_words,
            output_word_count=output_words,
            record_base=next_record_base,
            record_count=record_count,
            input_base_addr=0 if layer_index == 0 else ACTIVATION_BASE_ADDR + ((layer_index - 1) * ACTIVATION_REGION_BYTES),
            output_base_addr=ACTIVATION_BASE_ADDR + (layer_index * ACTIVATION_REGION_BYTES),
        )
        stages.append(stage)
        current_size = output_size
        previous_output_words = output_words
        next_record_base += record_count

    if next_record_base > MAX_CONV_RECORDS:
        raise ValueError(
            f"model needs {next_record_base} conv records, but current RTL supports "
            f"at most {MAX_CONV_RECORDS}"
        )
    return stages


def validate_ddr_layout(
    stages: list[ConvStage],
    conv_word_count: int,
    fc_word_count: int,
    fc_pair_count: int,
    class_count: int,
) -> None:
    """Reject exports whose fixed DDR regions or metadata fields would overlap."""
    if not 0 < fc_pair_count <= 0xFFFF:
        raise ValueError(f"FC input pair count {fc_pair_count} does not fit scheduler metadata")
    if not 0 < class_count <= 0xF:
        raise ValueError(f"FC class count {class_count} does not fit the 4-bit scheduler field")

    metadata_end = MODEL_METADATA_BASE_ADDR + (8 + MAX_CONV_LAYERS * 8) * 4
    conv_end = CONV_RECORD_BASE_ADDR + conv_word_count * 4
    fc_end = FC_WEIGHT_BASE_ADDR + fc_word_count * 4
    if metadata_end > CONV_RECORD_BASE_ADDR:
        raise ValueError("scheduler metadata overlaps the convolution-record DDR region")
    if conv_end > FC_WEIGHT_BASE_ADDR:
        raise ValueError(
            f"convolution records end at 0x{conv_end:08X}, overlapping the fixed "
            f"FC-weight base 0x{FC_WEIGHT_BASE_ADDR:08X}"
        )
    if fc_end > ACTIVATION_BASE_ADDR:
        raise ValueError(
            f"FC weights end at 0x{fc_end:08X}, overlapping the fixed activation "
            f"base 0x{ACTIVATION_BASE_ADDR:08X}"
        )

    for stage in stages:
        activation_bytes = stage.output_tiles * stage.output_word_count * 4
        if activation_bytes > ACTIVATION_REGION_BYTES:
            raise ValueError(
                f"{stage.name}: output activation region needs {activation_bytes} bytes, "
                f"exceeding {ACTIVATION_REGION_BYTES}"
            )


def export_conv_records(
    conv_pipeline: list[tuple[object, bool]],
    stages: list[ConvStage],
    scale: int,
    keep_conv_biases: bool,
) -> list[list[int]]:
    records: list[list[int]] = []

    for layer_index, ((layer, _), stage) in enumerate(zip(conv_pipeline, stages)):
        raw_weights, raw_biases = layer.get_weights()
        weights = quantize_signed16(raw_weights, scale)
        weights = customize_quantized_conv_weights(weights, layer_index, layer.name)
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

                    ram_dir = 1 if layer_index == 0 else (output_tile & 0x1)
                    pool_now = stage.use_pool and (input_tile == stage.input_tiles - 1)
                    control = control_word(
                        ram_dir=ram_dir,
                        img_size=stage.input_size,
                        use_pool=pool_now,
                        in_layer_size=int(math.log2(stage.offset_count)),
                        write_offset=offset,
                    )
                    records.append(make_record(kernels, record_biases, control))

    return records


def fc_ddr_feature_indices(final_stage: ConvStage) -> np.ndarray:
    """Map tile-major DDR activation order to Keras HWC flatten indices."""
    channels = final_stage.output_channels
    spatial_size = final_stage.output_size
    if channels % PARALLEL_CONV:
        raise ValueError("final convolution channel count must be a multiple of 8")

    keras_indices = np.arange(spatial_size * spatial_size * channels).reshape(
        spatial_size, spatial_size, channels
    )
    return keras_indices.reshape(
        spatial_size, spatial_size, channels // PARALLEL_CONV, PARALLEL_CONV
    ).transpose(2, 0, 1, 3).reshape(-1)


def export_fc_words(
    fc_layer: object,
    final_stage: ConvStage,
    scale: int,
) -> tuple[list[int], int, int]:
    weights, _ = fc_layer.get_weights()
    input_count, class_count = weights.shape
    if input_count % 2:
        raise ValueError("FC input count must be even because DDR stores two signed weights per word")

    feature_indices = fc_ddr_feature_indices(final_stage)
    if len(feature_indices) != input_count:
        raise ValueError(
            f"final activation feature count {len(feature_indices)} != FC input count {input_count}"
        )

    # Match tile-major DDR activation order.
    quantized = quantize_signed16(weights[feature_indices, :], scale)
    words: list[int] = []
    for class_index in range(class_count):
        words.extend(pack_signed16_pairs(quantized[:, class_index]))
    return words, input_count // 2, class_count


def export_metadata(stages: list[ConvStage], fc_pair_count: int, class_count: int) -> list[int]:
    words = [
        0x4D45_5441,
        len(stages),
        fc_pair_count,
        class_count,
        FC_WEIGHT_BASE_ADDR,
        CONV_RECORD_BASE_ADDR,
        0,
        0,
    ]

    for stage in stages:
        words.extend(
            [
                stage.output_tiles | (stage.input_tiles << 8) | (stage.offset_count << 16),
                stage.input_word_count,
                stage.output_word_count,
                stage.record_base,
                stage.input_base_addr,
                stage.output_base_addr,
                0,
                0,
            ]
        )

    while len(words) < 8 + (MAX_CONV_LAYERS * 8):
        words.append(0)
    return words


def write_tb_parameters(
    path: Path,
    *,
    stages: list[ConvStage],
    records: list[list[int]],
    fc_pair_count: int,
    class_count: int,
    metadata_path: Path,
    conv_mem_path: Path,
    fc_path: Path,
) -> None:
    fc_word_count = fc_pair_count * class_count
    conv_word_count = len(records) * RECORD_WORDS_32
    expected_replay_writes = sum(
        stage.output_tiles * stage.input_tiles * (stages[index - 1].output_word_count // 4)
        for index, stage in enumerate(stages)
        if index > 0
    )

    lines = [
        "Dense TOP_BIG_CNN_TB model parameters",
        "=====================================",
        "",
        "Generated by python/main_files/convert_to_hex_dense.py.",
        "The three .mem files and these constants describe one exported model.",
        "",
        "Model files",
        "-----------",
        f'BIG_MODEL_METADATA_FILE     = "{metadata_path.name}"',
        f'BIG_MODEL_CONV_RECORDS_FILE = "{conv_mem_path.name}"',
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
            "Common constants",
            "----------------",
            f"DDR_BASE_ADDR          = 32'h{MODEL_METADATA_BASE_ADDR:08X}",
            f"CONV_RECORD_BASE_ADDR  = 32'h{CONV_RECORD_BASE_ADDR:08X}",
            f"FC_WEIGHT_BASE_ADDR    = 32'h{FC_WEIGHT_BASE_ADDR:08X}",
            f"META_WORDS             = {8 + MAX_CONV_LAYERS * 8}",
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
        activation_bytes = stage.output_tiles * stage.output_word_count * 4
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
                f"  OUTPUT_ADDR_END      = 32'h{stage.output_base_addr + activation_bytes:08X} (exclusive)",
            ]
        )

    conv_end = CONV_RECORD_BASE_ADDR + conv_word_count * 4
    fc_end = FC_WEIGHT_BASE_ADDR + fc_word_count * 4
    lines.extend(
        [
            "",
            "DDR occupied address ranges",
            "---------------------------",
            f"Metadata:     32'h{MODEL_METADATA_BASE_ADDR:08X} .. 32'h{MODEL_METADATA_BASE_ADDR + (8 + MAX_CONV_LAYERS * 8) * 4:08X} (exclusive)",
            f"Conv records: 32'h{CONV_RECORD_BASE_ADDR:08X} .. 32'h{conv_end:08X} (exclusive)",
            f"FC weights:   32'h{FC_WEIGHT_BASE_ADDR:08X} .. 32'h{fc_end:08X} (exclusive)",
            "",
        ]
    )

    if len(stages) == 2:
        conv1, conv2 = stages
        lines.extend(
            [
                "Paste-ready block for the current two-conv TOP_BIG_CNN_TB.v",
                "---------------------------------------------------------",
                f"`define BIG_MODEL_CONV_RECORDS_FILE \"{conv_mem_path.as_posix()}\"",
                f"`define BIG_MODEL_METADATA_FILE \"{metadata_path.as_posix()}\"",
                f"`define BIG_MODEL_FC_WEIGHTS_FILE \"{fc_path.as_posix()}\"",
                "",
                f"localparam [31:0] DDR_BASE_ADDR         = 32'h{MODEL_METADATA_BASE_ADDR:08X};",
                f"localparam [31:0] CONV_RECORD_BASE_ADDR = 32'h{CONV_RECORD_BASE_ADDR:08X};",
                f"localparam [31:0] FC_WEIGHT_BASE_ADDR   = 32'h{FC_WEIGHT_BASE_ADDR:08X};",
                f"localparam [31:0] CONV1_ACT_BASE_ADDR   = 32'h{conv1.output_base_addr:08X};",
                f"localparam [31:0] CONV2_ACT_BASE_ADDR   = 32'h{conv2.output_base_addr:08X};",
                "",
                "localparam integer DDR_WORDS              = 262144;",
                f"localparam integer META_WORDS             = {8 + MAX_CONV_LAYERS * 8};",
                f"localparam integer RECORD_WORDS           = {RECORD_WORDS_32};",
                "localparam integer TILE_WORD_COUNT        = 4092;",
                f"localparam integer CONV1_TILE_WORD_COUNT  = {conv1.output_word_count};",
                f"localparam integer CONV2_TILE_WORD_COUNT  = {conv2.output_word_count};",
                "localparam integer LOCAL_TILE_ADDR_WIDTH  = 10;",
                f"localparam integer CONV1_TILES            = {conv1.output_tiles};",
                f"localparam integer CONV2_TILES            = {conv2.output_tiles};",
                f"localparam integer CONV2_INPUT_GROUPS     = {conv2.input_tiles};",
                f"localparam integer CONV2_OUTPUTS_PER_TILE = {conv2.offset_count};",
                f"localparam integer EXPECTED_CONV_RECORDS  = {len(records)};",
                f"localparam integer FC_INPUT_PAIR_COUNT    = {fc_pair_count};",
                f"localparam integer FC_CLASS_COUNT         = {class_count};",
                f"localparam integer FC_WEIGHT_WORD_COUNT   = {fc_word_count};",
                f"localparam integer EXPECTED_REPLAY_WRITES = {expected_replay_writes};",
            ]
        )
    else:
        lines.extend(
            [
                "Current testbench note",
                "----------------------",
                "The metadata-driven accelerator supports this exported model, but the current",
                "TOP_BIG_CNN_TB.v checker is explicitly written for two convolution layers.",
                "Generalize its CONV1/CONV2 activation scans before using this file unchanged.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    global FC_WEIGHT_BASE_ADDR

    args = parse_args()
    if args.fc_weight_base_addr is not None:
        FC_WEIGHT_BASE_ADDR = args.fc_weight_base_addr

    model_path = resolve_model_path(args.model)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.frac_bits <= 15:
        raise ValueError("--frac-bits must be between 0 and 15")
    scale = 1 << args.frac_bits

    model = load_model_compatible(model_path)
    conv_pipeline, fc_layer = keras_pipeline(model)
    stages = build_stages(conv_pipeline)
    records = export_conv_records(conv_pipeline, stages, scale, args.keep_conv_biases)
    fc_words, fc_pair_count, class_count = export_fc_words(fc_layer, stages[-1], scale)
    metadata = export_metadata(stages, fc_pair_count, class_count)

    final_feature_count = stages[-1].output_channels * stages[-1].output_size * stages[-1].output_size
    if final_feature_count != fc_pair_count * 2:
        raise ValueError(
            "final convolution output does not match FC input size: "
            f"{final_feature_count} != {fc_pair_count * 2}"
        )

    channel_suffix = "_".join(str(stage.output_channels) for stage in stages)
    prefix = args.prefix or f"model_conv{channel_suffix}"
    conv_mem_path = output_dir / f"{prefix}_records_dense.mem"
    metadata_path = output_dir / f"metadata_{prefix}_dense.mem"
    fc_path = output_dir / f"{prefix}_fc_weights_dense.mem"

    conv_words = pack_signed16_pairs(np.asarray(records, dtype=np.int16).reshape(-1))
    conv_storage_bytes = len(conv_words) * 4
    metadata_storage_bytes = len(metadata) * 4
    fc_storage_bytes = len(fc_words) * 4
    model_storage_bytes = conv_storage_bytes + metadata_storage_bytes + fc_storage_bytes
    validate_ddr_layout(stages, len(conv_words), len(fc_words), fc_pair_count, class_count)
    write_mem32(conv_mem_path, conv_words)
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
    print(f"FC weight words:    {len(fc_words)}")
    print(f"Model DDR bytes:    {model_storage_bytes}")
    print(f"Metadata:           {metadata_path}")
    print(f"Conv records:       {conv_mem_path}")
    print(f"FC weights:         {fc_path}")


if __name__ == "__main__":
    main()

# CUSTOM INT16 CONV WEIGHTS
# Put any of these lines inside customize_quantized_conv_weights() above.
# Valid values: -32768 to 32767.
# weights[:] = np.int16(123)                 # all Conv weights
# weights[0, 0, 0, 0] = np.int16(-456)      # one weight
# weights[:, :, :, 3] = np.int16(1000)       # output channel 3
# weights[:, :, 2, :] = np.int16(-25)        # input channel 2
