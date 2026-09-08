import time

from hello_fpga_driver import find_mcp2221_port, FpgaLink

IMG_W = 28
IMG_H = 28
ZERO_IMAGE = []
ALL_MAX_IMAGE = []
for i in range(IMG_W):
    r = []
    r2 = []
    for j in range(IMG_H):
        r.append(0)
        r2.append(255)
    ZERO_IMAGE.append(r)
    ALL_MAX_IMAGE.append(r2)


def await_ack(fpga_link: FpgaLink, expected_ack: str):
    exp_byte = ord(expected_ack)
    resp = ""
    while resp == "" or resp is None:
        resp = fpga_link.read_next_byte_from_m3()
        if resp != exp_byte and resp != "" and resp is not None:
            print("Unexpected response from FPGA: {}, hex: 0x{:02X}, ascii: {}".format(resp, resp, chr(resp)))
            resp = ""

def send_image_array(i, fpga_link: FpgaLink):
    # Implementation for sending image array
    fpga_link.send_text_to_m3("c")
    print("Sending image command to FPGA")
    for row in i:
        for pixel in row:
            fpga_link.send_bytes_to_m3([pixel]) # the upper half of the 16 bit is pixel
            await_ack(fpga_link, "i")
            fpga_link.send_bytes_to_m3([0])  # The upper half of the 16 bit is 0
            await_ack(fpga_link, "i")
    print("Sent image to FPGA")
    await_ack(fpga_link, "l")

def start_run_n_times(fpga_link: FpgaLink, n: int):
    n_low = n & 0xFF
    n_high = (n >> 8) & 0xFF
    fpga_link.send_text_to_m3("r")
    fpga_link.send_bytes_to_m3([n_low])
    fpga_link.send_bytes_to_m3([n_high])
    print("Sent run command to FPGA for {} times".format(n))
    await_ack(fpga_link, "k")
    print("Run command acknowledged by FPGA")
    await_ack(fpga_link, "d")
    # get the next char
    result = None
    while result is None:
        result = fpga_link.read_next_byte_from_m3()
    print("Received result from FPGA: {}".format(result))
    return result

if __name__ == "__main__":
    port = find_mcp2221_port()
    if port is None:
        print("MCP2221 not found")
        exit(1)
    print("Found MCP2221 on port {}".format(port))
    print("Connecting to FPGA")
    fpga_link = FpgaLink(port)
    fpga_link.connect()
    print("Connected to FPGA")

    await_ack(fpga_link, "s")
    await_ack(fpga_link, "A")
    await_ack(fpga_link, "2")
    await_ack(fpga_link, "3")
    await_ack(fpga_link, "y")

    send_image_array(ALL_MAX_IMAGE, fpga_link)

    time.sleep(1)  # Wait for a second before starting the run

    start_run_n_times(fpga_link, 1000)

    print("Disconnecting from FPGA")

    fpga_link.disconnect()