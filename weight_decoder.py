import numpy as np


def get_weights_as_32bit_int_array(file_path):
    with open(file_path, "r") as f:
        all = f.read()
        all = all.replace("\n", "").replace("\r", "").replace(" ", "").replace("u", "").replace("0x", "").replace("\t", "")
        w = all.split(",")
        # interpret as 32 bit unsigned integers hex
        w = [np.uint32(int(x, 16)) for x in w if x]
        return w

def make_bytes(weights_list):
    # convert to bytes
    weights_bytes = b""
    for w in weights_list:
        weights_bytes += w.tobytes()
    return weights_bytes


class DenseRecord:
    def __init__(self, weights, biases, control):
        self.weights = weights
        self.biases = biases
        self.control = control
        self.kernel = weights.reshape((8, 3, 3)) # 8 kernels of 3x3

"""
Dense has 260 records. Each dense record is 64 × 32 bits. 
the first 36 words has 8 complete 3×3 kernels (72 signed 16-bit weights), 
words 36–39 hold biases, word 40 holds control, and the rest is 0 padding.
"""
DENSE_RECORDS = 260
DENSE_RECORD_SIZE = 64 * 4 # 64 words of 32 bits each
def interpret_as_dense_kernels(weights_list_bytes):
    records = []
    for i in range(DENSE_RECORDS):
        start = i * DENSE_RECORD_SIZE
        end = start + DENSE_RECORD_SIZE
        record_bytes = weights_list_bytes[start:end]
        record_words = np.frombuffer(record_bytes, dtype=np.uint32)
        weights = record_words[:36]
        biases = record_words[36:40]
        control = record_words[40]
        records.append(DenseRecord(weights, biases, control))
    return records

        



if __name__ == "__main__":
    file_path = "weights/weights_dense.c"
    weights = get_weights_as_32bit_int_array(file_path)
    print(f"Read {len(weights)} weights from {file_path}.")
    weights = np.array(weights, dtype=np.int32)
    print(f"First 10 weights: {weights[:10]}")