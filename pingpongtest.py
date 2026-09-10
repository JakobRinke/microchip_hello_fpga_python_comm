from hello_fpga_driver import FpgaLink
from measure_pipeline import search_and_connect_to_fpga
import time
n = 1
fpga_link = search_and_connect_to_fpga(m3=True)  # Connect to FPGA without waiting for M3 start sequence
for i in range(20):
    present_time = time.time()
    for i in range(n):
        fpga_link.send_bytes_raw([ord("q")])
    postsent_time = time.time()
    for i in range(n):
        while fpga_link.read_raw_next_byte() != ord("a"):
            pass
    received_time = time.time()
    print(f"Round {i+1}: Round-trip time: {(received_time - present_time) * 1000:.3f} ms, Post-send delay: {(postsent_time - present_time) * 1000:.3f} ms")
    print("")