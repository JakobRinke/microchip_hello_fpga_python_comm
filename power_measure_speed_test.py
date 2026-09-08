from hello_fpga_driver import FpgaLink, POWER_REQUEST_BYTE, POWER_FRAME_CONST, POWER_FRAME_LEN
from measure_pipeline import measure_power_and_switch_n_times, search_and_connect_to_fpga


def pipelined_high_speed_power_measurement(fpga_link: FpgaLink, n: int, max_in_flight_requests: int = 2):
    # Start the pipelined measurement
    power_measurements = []
    sent_requests = 0
    received_measurements = 0
    current_buffer = bytearray()
    while received_measurements < n:
        if sent_requests < n and (sent_requests - received_measurements) < max_in_flight_requests:
            fpga_link.send_bytes_raw(
                [POWER_REQUEST_BYTE]*(max_in_flight_requests - (sent_requests-received_measurements))
            )
            sent_requests += (max_in_flight_requests - (sent_requests-received_measurements))
        byte_received = None
        while byte_received is None: byte_received = fpga_link.read_raw_next_byte()
        
        current_buffer.append(byte_received)
        if len(current_buffer) >= POWER_FRAME_LEN:
            if current_buffer[0] == POWER_REQUEST_BYTE:
                power_value = fpga_link.decode_power_from_frame(current_buffer)
                power_measurements.append(power_value)
                received_measurements += 1
            current_buffer = bytearray()  # Reset buffer for next frame
    return power_measurements





def measure_power_n_times(fpga_link: FpgaLink, n: int):
    power_measurements = []
    for _ in range(n):
        power = fpga_link.get_current_power()
        power_measurements.append(power)
    return power_measurements

def test_power_measurement_speed_fast(fpga_link: FpgaLink, n: int):
    import time
    start_time = time.time()
    power_measurements = measure_power_n_times(fpga_link, n)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return power_measurements, elapsed_time

def test_power_measurement_speed_with_switching(fpga_link: FpgaLink, n: int):
    import time
    start_time = time.time()
    power_measurements = measure_power_and_switch_n_times(fpga_link, n)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return power_measurements, elapsed_time

def test_power_measurement_speed_pipelining(fpga_link: FpgaLink, n: int, in_flight_requests: int = 2):
    import time
    while fpga_link.read_raw_next_byte() is not None:  # Clear any remaining bytes in the buffer
        pass
    fpga_link.get_current_power()  # Guarantee that the USB Mode is set to PIC32 rather than M3
    start_time = time.time()
    power_measurements = pipelined_high_speed_power_measurement(fpga_link, n, max_in_flight_requests=in_flight_requests)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return power_measurements, elapsed_time


if __name__ == "__main__":
    import time
    n = 100  # Number of power measurements to take
    fpga_link = search_and_connect_to_fpga()


    # Pipelined Measurement Test
    in_flight_requests = 2  # Number of in-flight requests for pipelining
    power_measurements, elapsed_time = test_power_measurement_speed_pipelining(fpga_link, n, in_flight_requests=in_flight_requests)
    print(f"Pipelined ({in_flight_requests} in-flight) measured power {n} times in {elapsed_time:.2f} seconds.")
    print(f"Speed of one pipelined power measurement: {elapsed_time / n * 1000:.6f} milliseconds.")
    print(f"Average power: {sum(power_measurements) / len(power_measurements):.5f} W")

    print("\n---\n")

    in_flight_requests = 4  # Number of in-flight requests for pipelining
    power_measurements, elapsed_time = test_power_measurement_speed_pipelining(fpga_link, n, in_flight_requests=in_flight_requests)
    print(f"Pipelined ({in_flight_requests} in-flight) measured power {n} times in {elapsed_time:.2f} seconds.")
    print(f"Speed of one pipelined power measurement: {elapsed_time / n * 1000:.6f} milliseconds.")
    print(f"Average power: {sum(power_measurements) / len(power_measurements):.5f} W")

    print("\n---\n")

    # Fast Measurement Test
    power_measurements, elapsed_time = test_power_measurement_speed_fast(fpga_link, n)
    print(f"Measured power {n} times in {elapsed_time:.2f} seconds.")
    print(f"Speed of one power measurement: {elapsed_time / n * 1000:.6f} milliseconds.")
    print(f"Average power: {sum(power_measurements) / len(power_measurements):.5f} W")

    print("\n---\n")

    power_measurements, elapsed_time = test_power_measurement_speed_with_switching(fpga_link, n)
    print(f"Measured power {n} times with switching in {elapsed_time:.2f} seconds.")
    print(f"Speed of one power measurement with switching: {elapsed_time / n * 1000:.6f} milliseconds.")
    print(f"Average power: {sum(power_measurements) / len(power_measurements):.5f} W")