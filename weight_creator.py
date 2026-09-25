from pathlib import Path
import numpy as np

import convert_to_hex_dense as cvt
import sparse_hex_file as sp


def constant_weights_dense(weights, layer_index, layer_name, const_value):
    weights[:] = np.int16(const_value)
    return weights


def constant_weights_sparse(weights, layer_index, layer_name, const_value):
    weights[weights != 0] = np.int16(const_value)
    return weights

def create_constant_weight_stream_dense(const_value:np.int16=256):
    cvt.customize_quantized_conv_weights = (
        lambda weights, layer_index, layer_name: constant_weights_dense(
            weights, layer_index, layer_name, const_value
        )
    )

    model_path = cvt.resolve_model_path(Path("models/mnist_cnn_32_64.h5"))
    model = cvt.load_model_compatible(model_path)
    conv_pipeline, fc_layer = cvt.keras_pipeline(model)
    stages = cvt.build_stages(conv_pipeline)

    scale = 1 << 14
    records = cvt.export_conv_records(
        conv_pipeline, stages, scale, keep_conv_biases=False
    )
    conv_words = cvt.pack_signed16_pairs(
        np.asarray(records, dtype=np.int16).reshape(-1)
    )
    return np.array(conv_words, dtype=np.uint32)


def create_constant_weight_stream_sparse(const_value:np.int16=256):
    sp.customize_quantized_sparse_weights = (
        lambda weights, layer_index, layer_name: constant_weights_sparse(
            weights, layer_index, layer_name, const_value
        )
    )

    model_path = sp.resolve_model_path(Path("models/mnist_cnn_32_64_sparse.h5"))
    model = sp.load_model_compatible(model_path)
    conv_pipeline, fc_layer = sp.keras_pipeline(model)
    stages = sp.build_stages(conv_pipeline)

    scale = 1 << 14
    records = sp.export_conv_records(
        conv_pipeline, stages, scale, keep_conv_biases=False
    )
    conv_words = [word for record in records for word in sp.record_words(record)]
    return np.array(conv_words, dtype=np.uint32)