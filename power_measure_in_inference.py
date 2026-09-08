import os
import json
import time
import random
from hello_fpga_driver import FpgaLink, SWITCH_PIC_CMD
from data_helpers import *
from measure_pipeline import search_and_connect_to_fpga, start_run_n_times, imgs, \
                                send_image_array, await_ack, GOT_RUN_CMD_ACK, RUN_COMPLETE_ACK


NUM_OF_CHECKS_PER_IMG = 20
NUM_OF_MEAS_PER_INFERENCE = 15
MAX_RANDOM_START_DELAY = 0.028 # 28 ms 
OUTPUT_FILE = "in_inference_power_25hz.json"
CONST_SWITCH_DELAY = 0.06 # 60ms, no more so all inference power measures are in sync

def measure_power_by_time_with_random_delay(fpga_link: FpgaLink, n: int, max_random_delay: float = MAX_RANDOM_START_DELAY):
    power_measurements = {}
    # Manual Switch (unsafe, but we need the speed)
    bt = time.time()
    fpga_link._switch_to_pic() # this takes 50 ms, we assume that the FPGA waits at least 50 ms to let us switch
    switch_delay = time.time() - bt
    time.sleep(CONST_SWITCH_DELAY - switch_delay) # Wait a bit to make sure the FPGA is in inference mode
    start_time = time.time()
    time.sleep(random.uniform(0, max_random_delay)) # Random delay to avoid synchronization with the FPGA
    for _ in range(n):
        mes_s = time.time()
        power = fpga_link.get_current_power()
        mes_e = time.time()
        # Calculate the timestamp - Assumption - power gets measured at the middle of the measurement time
        timestamp = (mes_s - start_time) + (mes_e - mes_s) / 2
        power_measurements[timestamp] = power
    return power_measurements

def measure_power_for_image(fpga_link: FpgaLink, image_array, n_runs, max_random_delay: float = MAX_RANDOM_START_DELAY):
    send_image_array(image_array, fpga_link)
    power_mes = {}
    # clear the fpga buffer before starting the measurements
    for _ in range(n_runs):
        start_run_n_times(fpga_link, 1, do_await_ack=False)
        power_mes.update(measure_power_by_time_with_random_delay(fpga_link, NUM_OF_MEAS_PER_INFERENCE, max_random_delay))
        await_ack(fpga_link, GOT_RUN_CMD_ACK)
        await_ack(fpga_link, RUN_COMPLETE_ACK)
        fpga_link.read_next_byte_from_m3() # get the result, don't care about it, just to clear the buffer
        time.sleep(0.2) # Wait a bit before the next run to avoid overlapping measurements
    # order dict by timestamp
    power_mes = dict(sorted(power_mes.items()))
    return power_mes


if __name__ == "__main__":
    output_file = input(f"Enter the output file name (default: {OUTPUT_FILE}): ")
    if not output_file:
        output_file = OUTPUT_FILE

    already_done_runs = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            already_done_runs = json.load(f)

    fpga_link = search_and_connect_to_fpga()

    for i, image_array in enumerate(imgs):

        if len(already_done_runs) >= i + 1:
            print(f"Run for image_{i} already done. Skipping.")
            continue
        print(f"Measuring power for image_{i}...")
        power_measurements = measure_power_for_image(fpga_link, image_array, NUM_OF_CHECKS_PER_IMG)
        already_done_runs.append(power_measurements)
        with open(output_file, "w") as f:
            json.dump(already_done_runs, f, indent=4)