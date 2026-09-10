from PIL import Image
import numpy as np
from hello_fpga_driver import find_mcp2221_port, FpgaLink
import json
import os
import time
from data_helpers import *



START_SEQUENCE = ["s", "A", "2", "3", "y"]

SEND_IMAGE_CMD = "c"
RUN_CMD = "r"

GOT_IMAGE_BYTE_ACK = "i"
LOADED_IMAGE_ACK = "l"
GOT_RUN_CMD_ACK = "k"
RUN_COMPLETE_ACK = "d"


TEST_IDLE_NUM = 100
TEST_RUN_NUM = 1000

output_file = "power_measurements.json"

def done_run(name, already_done_runs):
    for i in already_done_runs:
        if i["name"] == name:
            return True
    return False

def load_runs():
    global output_file
    already_done_runs = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            already_done_runs = json.load(f)
    return already_done_runs

def save_runs(results):
    global output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def load_image_as_array(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img_array = np.array(img)
    return img_array

def load_all_images_from_directory(directory_path):
    image_arrays = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory_path, filename)
            img_array = load_image_as_array(image_path)
            image_arrays.append(img_array)
    return image_arrays

def search_and_connect_to_fpga(m3=True):
    port = find_mcp2221_port()
    if port is None:
        raise Exception("No MCP2221 device found")
    print(f"Found MCP2221 device on port: {port}")
    fpga_link = FpgaLink(port)
    print("Connecting to FPGA...")
    fpga_link.connect()
    print("Connected to FPGA")
    if m3:
        print("Waiting for M3 start sequence...")
        for expected_ack in START_SEQUENCE:
            await_ack(fpga_link, expected_ack)
        print("M3 start sequence received")
    return fpga_link

def await_ack(fpga_link: FpgaLink, expected_ack: str, silently=False):
    exp_byte = ord(expected_ack)
    resp = ""
    while resp == "" or resp is None:
        resp = fpga_link.read_next_byte_from_m3()
        if resp != exp_byte and resp != "" and resp is not None:
            if not silently:
                print("Unexpected response from FPGA: {}, hex: 0x{:02X}, ascii: {}".format(resp, resp, chr(resp)))
            resp = ""

def await_ack_with_power_measure(fpga_link: FpgaLink, expected_ack: str, drop_last=True):
    exp_byte = ord(expected_ack)
    resp = ""
    power_measurements = []
    while resp == "" or resp is None:
        p = fpga_link.get_current_power()
        power_measurements.append(p)
        resp = fpga_link.read_next_byte_from_m3()
        if resp != exp_byte and resp != "" and resp is not None:
            print("Unexpected response from FPGA: {}, hex: 0x{:02X}, ascii: {}".format(resp, resp, chr(resp)))
            resp = ""
    if drop_last:
        power_measurements = power_measurements[:-1]
    return power_measurements

def measure_power_and_switch_n_times(fpga_link: FpgaLink, n: int):
    power_measurements = []
    for _ in range(n):
        p = fpga_link.get_current_power()
        power_measurements.append(p)
        resp = fpga_link.read_next_byte_from_m3()
        if resp != None:
            print("Unexpected response from FPGA: {}, hex: 0x{:02X}, ascii: {}".format(resp, resp, chr(resp)))
    return power_measurements

def send_image_array(i, fpga_link: FpgaLink):
    # Implementation for sending image array
    fpga_link.send_text_to_m3(SEND_IMAGE_CMD)
    for row in i:
        for pixel in row:
            fpga_link.send_bytes_to_m3([pixel]) # the upper half of the 16 bit is pixel
            await_ack(fpga_link, GOT_IMAGE_BYTE_ACK)
            fpga_link.send_bytes_to_m3([0])  # The upper half of the 16 bit is 0
            await_ack(fpga_link, GOT_IMAGE_BYTE_ACK)
    await_ack(fpga_link, LOADED_IMAGE_ACK)


def start_run_n_times(fpga_link: FpgaLink, n: int, do_await_ack=True):
    n_low = n & 0xFF
    n_high = (n >> 8) & 0xFF
    fpga_link.send_text_to_m3(RUN_CMD)
    fpga_link.send_bytes_to_m3([n_low])
    fpga_link.send_bytes_to_m3([n_high])
    if do_await_ack:
        await_ack(fpga_link, GOT_RUN_CMD_ACK)

def get_run_result_after_ack(fpga_link: FpgaLink):
    result = None
    while result is None:
        result = fpga_link.read_next_byte_from_m3()
    return result

def log_complete_run(name, average_power, standard_deviation, confidence_interval, min_power, max_power, power_measure_freq, result, power_measurements, elapsed_time):
    print(f"Run complete for {name}. Average power: {average_power:.4f} W, Standard deviation: {standard_deviation:.4f} W, Confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
    print(f"Min power: {min_power:.4f} W, Max power: {max_power:.4f} W")
    print(f"Power measurement frequency: {power_measure_freq:.2f} Hz")
    print(f"Result from FPGA: {result}")
    print(f"Elapsed time for run: {elapsed_time:.2f} seconds")
    print(f"------------------------------------------------")
    

def exec_run(name, fpga_link: FpgaLink, image_array, n_runs):
    print()
    print(f"Starting run for {name} with {n_runs} runs")
    send_image_array(image_array, fpga_link)
    start_run_n_times(fpga_link, n_runs)
    start_time = time.time()
    power_measurements = await_ack_with_power_measure(fpga_link, RUN_COMPLETE_ACK)
    end_time = time.time()
    elapsed_time = end_time - start_time

    average_power = sum(power_measurements) / len(power_measurements)
    standard_deviation = get_std_deviation(power_measurements)
    confidence_interval = get_95_confidence_interval(power_measurements)
    min_power, max_power = get_min_max(power_measurements)
    power_meaure_freq = len(power_measurements) / elapsed_time
    result = get_run_result_after_ack(fpga_link)
    
    log_complete_run(name, average_power, standard_deviation, confidence_interval, min_power, max_power, power_meaure_freq, result, power_measurements, elapsed_time)

    return {
        "name": name,
        "run_number": n_runs,
        "standard_deviation": standard_deviation,
        "confidence_interval": {
            "lower": confidence_interval[0],
            "upper": confidence_interval[1]
        },
        "min_power": min_power,
        "max_power": max_power,
        "power_measure_freq": power_meaure_freq,
        "average_power": average_power,
        "result": result,
        "power_measurements": power_measurements,
        "elapsed_time": elapsed_time
    }

def measure_idle_run(name, fpga_link: FpgaLink, n_runs):
    print()
    print(f"Starting idle run for {name} with {n_runs} runs")
    start_time = time.time()
    power_measurements = measure_power_and_switch_n_times(fpga_link, n_runs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    average_power = get_average(power_measurements)
    standard_deviation = get_std_deviation(power_measurements)
    confidence_interval = get_95_confidence_interval(power_measurements)
    min_power, max_power = get_min_max(power_measurements)
    power_meaure_freq = len(power_measurements) / elapsed_time
    result = 0
    log_complete_run(name, average_power, standard_deviation, confidence_interval, min_power, max_power, power_meaure_freq, result, power_measurements, elapsed_time)

    return {
        "name": name,
        "run_number": 0,
        "average_power": average_power,
        "standard_deviation": get_std_deviation(power_measurements),
        "confidence_interval": {
            "lower": confidence_interval[0],
            "upper": confidence_interval[1]
        },
        "min_power": min_power,
        "max_power": max_power,
        "power_measure_freq": power_meaure_freq,
        "result": result,
        "power_measurements": power_measurements,
        "elapsed_time": elapsed_time
    }

def do_run_if_not_done(name, fpga_link: FpgaLink, image_array, n_runs, already_done_runs):
    if done_run(name, already_done_runs):
        print(f"Run for {name} already done. Skipping.")
        return already_done_runs
    else:
        out = exec_run(name, fpga_link, image_array, n_runs)
        already_done_runs.append(out)
        save_runs(already_done_runs)
        return already_done_runs

def do_idle_run_if_not_done(name, fpga_link: FpgaLink, n_runs, already_done_runs):
    if done_run(name, already_done_runs):
        print(f"Idle run for {name} already done. Skipping.")
        return already_done_runs
    else:
        out = measure_idle_run(name, fpga_link, n_runs)
        already_done_runs.append(out)
        save_runs(already_done_runs)
        return already_done_runs




ALL_ZERO_IMAGE = np.zeros((28, 28), dtype=np.uint8)
ALL_MAX_IMAGE = np.full((28, 28), 255, dtype=np.uint8)
ALL_MIDDLE_IMAGE = np.full((28, 28), 128, dtype=np.uint8)

IMAGE_DIRECTORY = "./test_data/png_28x28"
imgs = load_all_images_from_directory(IMAGE_DIRECTORY)

if __name__ == "__main__":
    output_file = input("Enter the output file name (default: power_measurements.json): ")
    if not output_file:
        output_file = "power_measurements.json"

    already_done_runs = load_runs()

    fpga_link = search_and_connect_to_fpga()
    # Measure idle run
    already_done_runs = do_idle_run_if_not_done("idle", fpga_link, TEST_IDLE_NUM, already_done_runs)

    # Measure runs for all zero image
    already_done_runs = do_run_if_not_done("all_zero", fpga_link, ALL_ZERO_IMAGE, TEST_RUN_NUM, already_done_runs)
    # Measure runs for all max image
    already_done_runs = do_run_if_not_done("all_max", fpga_link, ALL_MAX_IMAGE, TEST_RUN_NUM, already_done_runs)
    # Measure runs for all middle image
    already_done_runs = do_run_if_not_done("all_middle", fpga_link, ALL_MIDDLE_IMAGE, TEST_RUN_NUM, already_done_runs)

    # Measure runs for all images in the directory
    for idx, img_array in enumerate(imgs):
        name = f"image_{idx}"
        already_done_runs = do_run_if_not_done(name, fpga_link, img_array, TEST_RUN_NUM, already_done_runs)
    