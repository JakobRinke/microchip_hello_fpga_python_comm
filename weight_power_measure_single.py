from power_measure_in_inference import search_and_connect_to_fpga, measure_power_for_image, FpgaLink, imgs, await_ack, NUM_OF_CHECKS_PER_IMG
from single_conv import create_constant_weight_stream_dense
import json
import os

SEND_WEIGHT_CMD = "w"
GOT_WEIGHT_BYTE_ACK = "o"
LOADED_IMAGE_ACK = "l"

OUTPUT_FILE_NAME = "results/single_dense_power_prune_1/weights_{idx}.json"
if not os.path.exists("/".join(OUTPUT_FILE_NAME.split("/")[:-1])):
    os.makedirs("/".join(OUTPUT_FILE_NAME.split("/")[:-1]), exist_ok=True)

def read_output_file(file_name: str) -> list:
    if not os.path.exists(file_name):
        return []
    with open(file_name, "r") as f:
        return json.load(f)

def store_output_file(file_name: str, data: list):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

def add_result_to_output_file(file_name: str, result: dict):
    data = read_output_file(file_name)
    data.append(result)
    store_output_file(file_name, data)

def output_file_has_result(file_name: str, image_index):
    data = read_output_file(file_name)
    if len(data) > image_index:
        return True
    return False

def send_weight_stream_to_fpga(fpga_link: FpgaLink, weight_stream):
    fpga_link.send_text_to_m3(SEND_WEIGHT_CMD)
    for weight in weight_stream:
        w0 = weight & 0xFF
        w1 = (weight >> 8) & 0xFF
        w2 = (weight >> 16) & 0xFF
        w3 = (weight >> 24) & 0xFF
        fpga_link.send_bytes_to_m3(bytes([w0, w1, w2, w3]))
        await_ack(fpga_link, GOT_WEIGHT_BYTE_ACK)
    await_ack(fpga_link, LOADED_IMAGE_ACK)
    print("Weight stream sent to FPGA successfully.")


def process_const_weights(fpga_link: FpgaLink, weight_value: int):
    if weight_value >= 0x8000:
        weight_real = weight_value - 0x10000  # Convert to signed 16-bit integer
    else:
        weight_real = weight_value
    print(f"Creating constant weight stream for value: {weight_real}")
    weight_stream = create_constant_weight_stream_dense(weight_real, out_channels=128, num_classes=1)
    print("Weight stream: ", weight_stream[:10], "...")  # Print first 10 weights for verification
    send_weight_stream_to_fpga(fpga_link, weight_stream)
 
    output_file_name = OUTPUT_FILE_NAME.format(idx=hex(weight_value))
    for idx, img in enumerate(imgs):
        if output_file_has_result(output_file_name, idx):
            print(f"Result for image {idx} already exists. Skipping.")
            continue
        print(f"Measuring power for image {idx} with constant weight {weight_value}.")
        result = measure_power_for_image(fpga_link, img, NUM_OF_CHECKS_PER_IMG)
        add_result_to_output_file(output_file_name, result)
        print(f"Result for image {idx} stored in {output_file_name}.")



NUM_OF_CHECKS_PER_IMG = 2

WEIGHTS_TO_TEST = [
    0,  # All weights are zero
    0x7FFF,  # All weights are max positive
    0xFFFF,  # All ones
    0x8000,  # All weights are max negative -> Also Bit Shift
    0x2000,  # Simple Bit Shift for Multiplication
    0b1010101010101010,  # Alternating bits
    0b0101010101010101,  # Alternating bits
    0x4000,  # Another bit shift value
    0x1234,  # Random weight value
    0x1F2E,  # Random weight value
    0x6312,  # Random weight value
    0x7A5C,  # Random weight value
    0xE4D3   # Random weight value
]

if __name__ == "__main__":
    fpga_link = search_and_connect_to_fpga()
    for weight_value in WEIGHTS_TO_TEST:
        print(f"Processing constant weight value: {weight_value}")
        process_const_weights(fpga_link, weight_value)