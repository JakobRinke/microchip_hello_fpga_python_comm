from hello_fpga_driver import FpgaLink, find_and_connect_to_fpga
import time
import random
import json

SEND_IMAGE_CMD = "c"
RUN_CMD = "r"
SEND_WEIGHT_CMD = "w"


GOT_IMAGE_BYTE_ACK = "i"
GOT_WEIGHT_BYTE_ACK = "o"
LOADED_IMAGE_ACK = "l"
GOT_RUN_CMD_ACK = "k"
RUN_COMPLETE_ACK = "d"

MAX_RANDOM_START_DELAY = 0.022
CONST_SWITCH_DELAY = 0.2

OVERSAMPLING_SIZE = 1
POWER_MEASUREMENTS_PER_INFERENCE = 80
RUNS_IN_INFERENCE = 40

def send_image_array(fpga_link: FpgaLink, image_array):
    """ Sends an image array to the FPGA. The image_array should be a 2D list of pixel 8-bit values. """
    fpga_link.send_text_to_m3(SEND_IMAGE_CMD)
    print("Sending image command to FPGA")
    i = 0
    for row in image_array:
        for pixel in row:
            fpga_link.send_bytes_to_m3([pixel]) # the upper half of the 16 bit is pixel
            fpga_link.await_ack(GOT_IMAGE_BYTE_ACK)
            fpga_link.send_bytes_to_m3([0])  # The upper half of the 16 bit is 0
            fpga_link.await_ack(GOT_IMAGE_BYTE_ACK)
    print("Sent image to FPGA")
    fpga_link.await_ack(LOADED_IMAGE_ACK)

def send_weight_stream(fpga_link: FpgaLink, weight_stream):
    """ Sends a stream of weights to the FPGA. The weight_stream should be an iterable of a 32-bit ainteger weights stream. """
    fpga_link.send_text_to_m3(SEND_WEIGHT_CMD)
    for weight in weight_stream:
        w0 = weight & 0xFF
        w1 = (weight >> 8) & 0xFF
        w2 = (weight >> 16) & 0xFF
        w3 = (weight >> 24) & 0xFF
        fpga_link.send_bytes_to_m3(bytes([w0, w1, w2, w3]))
        fpga_link.await_ack(GOT_WEIGHT_BYTE_ACK)
    fpga_link.await_ack(LOADED_IMAGE_ACK)
    print("Weight stream sent to FPGA successfully.")

def start_run_n_times(fpga_link: FpgaLink, n: int, await_started_ack=True):
    """ Starts the CNN run on the FPGA for n times. If await_started_ack is True, it will wait for the FPGA to acknowledge that the run has started. """
    n_low = n & 0xFF
    n_high = (n >> 8) & 0xFF
    fpga_link.send_text_to_m3(RUN_CMD)
    fpga_link.send_bytes_to_m3([n_low])
    fpga_link.send_bytes_to_m3([n_high])
    if await_started_ack:
        fpga_link.await_ack(GOT_RUN_CMD_ACK)

def await_run_start(fpga_link: FpgaLink):
    """ Waits for the FPGA to acknowledge that the CNN Run has started. """
    fpga_link.await_ack(GOT_RUN_CMD_ACK)

def await_run_end(fpga_link: FpgaLink) -> int:
    """ Waits for the CNN Run to complete and returns the result sent by the FPGA. """
    fpga_link.await_ack(RUN_COMPLETE_ACK)
    time.sleep(0.1)
    b = None
    while b is None:
        b = fpga_link.read_next_byte_from_m3()
    return b

def measure_power_by_time(
        fpga_link: FpgaLink, 
        num_measurements: int, 
        oversampling_delay_range: float = MAX_RANDOM_START_DELAY,
        before_measurement_delay: float = CONST_SWITCH_DELAY
    ):
    """ Measures power consumption over time while the FPGA is in inference mode. Returns a dictionary mapping timestamps to power measurements. """
    power_measurements = {}
    bt = time.time()
    fpga_link._switch_to_pic()
    switch_delay = time.time() - bt
    time.sleep(before_measurement_delay - switch_delay) # Wait a bit to make sure the FPGA is in inference mode
    start_time = time.time()
    time.sleep(random.uniform(0, oversampling_delay_range)) # Random delay to avoid synchronization with the FPGA (oversampling)
    for _ in range(num_measurements):
        mes_s = time.time()
        power = fpga_link.get_current_power()
        mes_e = time.time()
        # Calculate the timestamp - Assumption - power gets measured at the middle of the measurement time
        timestamp = (mes_s - start_time) + (mes_e - mes_s) / 2
        power_measurements[timestamp] = power
    return power_measurements

def run_power_measurement_for_image(
    fpga_link: FpgaLink,
    image_array,
    runs_per_inference: int = RUNS_IN_INFERENCE,
    power_measurements_per_inference: int = POWER_MEASUREMENTS_PER_INFERENCE,
    oversampling_size: int = OVERSAMPLING_SIZE,
    oversampling_delay_range: float = MAX_RANDOM_START_DELAY,
    before_measurement_delay: float = CONST_SWITCH_DELAY
) -> dict:
    """ Sends an image to the FPGA, runs inference multiple times, and measures power consumption during each run. Returns a dictionary mapping timestamps to power measurements. """
    send_image_array(fpga_link, image_array)
    power_mes = {}
    # clear the fpga buffer before starting the measurements
    for _ in range(oversampling_size):
        start_run_n_times(fpga_link, runs_per_inference, await_started_ack=False)
        power_mes.update(measure_power_by_time(fpga_link, power_measurements_per_inference, oversampling_delay_range, before_measurement_delay))
        await_run_start(fpga_link)
        await_run_end(fpga_link)
    # order dict by timestamp
    power_mes = dict(sorted(power_mes.items()))
    return power_mes

CONNECTION_START_SEQUENCE = ["s", "A", "2", "3", "y"]
def search_and_connect_to_fpga_and_init_m3(start_seq=CONNECTION_START_SEQUENCE) -> FpgaLink:
    fpga_link = find_and_connect_to_fpga()
    for ack in start_seq:
        fpga_link.await_ack(ack)
    print("M3 initialization complete.")
    return fpga_link


if __name__ == "__main__":
    fpga_link = search_and_connect_to_fpga_and_init_m3()
    # Don't send a constant steam since it is missing the configuration bytes!!!
    # random_weight_stream = [0] * 1344  
    # send_weight_stream(fpga_link, random_weight_stream)
    random_image = [[0 for _ in range(28)] for _ in range(28)]
    power_measurements = run_power_measurement_for_image(fpga_link, random_image)
    print("Power measurements:", json.dumps(power_measurements, indent=2))