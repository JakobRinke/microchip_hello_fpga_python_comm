#!/usr/bin/env python3
import sys
import time
import threading
import argparse
import base64
from collections import deque
import logging
try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing. Please use: pip install pyserial")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---- MCP2221 Identifikation ----
MCP2221_VID = 0x04D8
MCP2221_PID = 0x00DD

# ---- Power-Frame-Definition (reverse-engineered) ----
POWER_REQUEST_BYTE = 0x2A          # '*'
POWER_FRAME_LEN = 8                # incl. Echo-Byte
POWER_FRAME_CONST = {
    3: 0x07,
    4: 0x80,
    5: 0x58,
    6: 0x5D,
    7: 0x61,
}
FRAME_BYTE_TIMEOUT = 0.3 
POWER_SCALE = 0.00011724
RAW_FLUSH_IDLE = 0.08
RAW_FLUSH_MAXLEN = 64

# ---- Boot-/DirectC-Meilensteine after identification ----
POST_ID_MILESTONES = [
    b"ActID",
    b"FPGA Array is programmed and enabled",
    b"Design Name",
    b"Bitstream NVM Digest",
    b"Exit code",
    b"Security locks",
    b"End DirectC Demo",
]

# Amount of delay before first messages are sent
DEFAULT_SETTLE_DELAY = 3.0

SAFETY_SWITCH_DELAY = 0.0005  # Delay after switching to M3 before sending the first byte

# ---- Identificaion Block-Block ----
_ID_BLOB_B64 = (
    "RzRNLURlc2lnbmVyICAgICAgICAgICAgRpINCQAEAAAAAAAABc8xgA////8PqgAAAAFsygIGAAUABAAA////D6oAAAABbMoCBgAF"
    "AAEAAAAAAAAAAAAAAAYCgAAAAAkAAAADkAAAAFoAAAAE8AAAAFoAAAAFUAEAABEAAAAIcAEAACAMCQAJkAsJAAAAAAAAAAAAhXxF"
    "IBAIBAIBAAAAAAAAACRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
    "kiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAwBYAkKQIFBcAGQAACAAAAAAA"
    "AAAAAAAAAAAAAAAA+z+ZBGd8kdgx7hYXbsV+WzpzIJ4DSJEI2nsR4nGWjG7nX0/HOCEaElZLeJz/8xgqAAMAAgAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+P//AxMAwAFsVE9QAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAACZI2ljqbjdTbkY1/lkbE7FYzoqintZkQXwqrQbb9QElm9EEWmun"
    "xdPCob12uakI7rF/X5p6YlRF20xkFGEz+lN7QZoHiimrI9k8jOIIb6+RpgTcRnrJOEXmVoeMJhn/zehzCMlmN4tV3EO/Udr0nfFV"
    "n27BuTI5WuHciBycinovb5JdFyBRBaAh4x9HBrGb/fBxlKb2TteVKJ8aW9ZiK+0CAwACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAFzzGAD////w+qAAAAAWzKAgYABQABAAAAAAAAAAAAAAAGAoAA"
    "AAAJAAAAA5AAAABaAAAABPAAAABaAAAABVABAAARAAAACHABAAAgDAkACZALCQAAAAAAAAAAAIV8RSAQCAQCAQAAAAAAAAAkSZIk"
    "SZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIk"
    "SZIkSZIkSZIkSQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAMAWAJCkCBQXABkAAAgAAAAAAAAAAAAAAAAAAAAAAPs/mQRn"
    "fJHYMe4WF27Ffls6cyCeA0iRCNp7EeJxloxu519PxzghGhJWS3ic//MYKgADAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/j//wMTAMABbFRPUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAQAAAAAAAAAAAAAAAAAAAAAAAAmSNpY6m43U25GNf5ZGxOxWM6Kop7WZEF8Kq0G2/UBJZvRBFprp8XTwqG9drmpCO6xf1+aemJU"
    "RdtMZBRhM/pTe0GaB4opqyPZPIziCG+vkaYE3EZ6yThF5laHjCYZ/83ocwjJZjeLVdxDv1Ha9J3xVZ9uwbkyOVrh3IgcnIp6L2+S"
    "XRcgUQWgIeMfRwaxm/3wcZSm9k7XlSifGlvWYivtAgMAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAA"
)
ID_BLOB = base64.b64decode(_ID_BLOB_B64)
ID_BLOB_HEADER_LEN = 56


SWITCH_M3_CMD = ord('&')  # '&'
SWITCH_PIC_CMD = [0x25, 0x45, 0x78, 0x49, 0x74, 0x33] # '%ExIt3'


STATE_DISCONNECTED = 0
STATE_PIC = 1
STATE_M3 = 2

def _split_id_blob(blob: bytes, header_len: int = ID_BLOB_HEADER_LEN):
    header = blob[:header_len]
    rest = blob[header_len:]
    chunks = []
    pos = 0
    while pos < len(rest):
        length = int.from_bytes(rest[pos:pos + 4], "little")
        pos += 4
        chunk = rest[pos:pos + length]
        pos += length
        chunks.append((length, chunk))
    return header, chunks


def find_mcp2221_port():
    """Searches for MCP2221 automatically."""
    for p in list_ports.comports():
        if p.vid == MCP2221_VID and p.pid == MCP2221_PID:
            return p.device
    return None


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def ascii_repr(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


class FpgaLink:
    """ THIS CLASS IS NOT THREAD-SAFE. It is the caller's responsibility to ensure that only one thread accesses the class at a time. """
    def __init__(self, port: str, baudrate: int = 460800, timeout: float = 0.05,
                  settle_delay: float = DEFAULT_SETTLE_DELAY):
        self.ser = serial.Serial(port, baudrate=baudrate, bytesize=8,
                                  parity='N', stopbits=1, timeout=timeout)
        self.settle_delay = settle_delay
        self.power_ready = False
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._raw_buf = bytearray()
        self._last_raw_time = 0.0
        self._pending = bytearray()
        self._pending_type = None
        self._pending_start_time = 0.0
        self._state = STATE_DISCONNECTED
        self.m3_rx_queue = deque()

    def start(self):
        self._reader_thread.start()

    def disconnect(self, close_port: bool = False):
        logger.info("Disconnecting ...")
        self.power_ready = False
        self._state = STATE_DISCONNECTED
        if close_port:
            self.ser.close()
        logger.info("Disconnected.")

    # ---------- Senden ----------
    def send_bytes_raw(self, data: bytes):
        logger.debug(f"[TX  ] hex={hexdump(data)}")
        logger.debug(f"       ascii='{ascii_repr(data)}'")
        self.ser.write(data)

    def send_text_raw(self, text: str):
        self.send_bytes_raw(text.encode("ascii", errors="replace"))

    def _request_power_raw(self):
        """ Sends a power request message (0x2A) to the board. """
        if not self._state == STATE_PIC:
            logger.error("Board is not in the PIC state - cannot request power. Please switch to PIC first.")
            return
        if not self.power_ready:
            logger.warning("Board is not ready for power requests yet (Connect-Handshake not completed or still in boot sequence).")
        self.send_bytes_raw(bytes([POWER_REQUEST_BYTE]))

    def connect(self, poll_timeout: float = 2.0, poll_interval: float = 1.4,
                max_tries: int = 8):
        logger.info("Starting connect handshake ...")
        self.ser.reset_input_buffer()
        self.power_ready = False
        got_echo = False
        for attempt in range(1, max_tries + 1):
            self.send_bytes_raw(b'#')
            start = time.time()
            while time.time() - start < poll_timeout:
                resp = self.ser.read(1)
                if resp:
                    logger.debug(f"[RX  ] hex={hexdump(resp)}  ascii='{ascii_repr(resp)}'")
                    if resp == b'#':
                        got_echo = True
                    break
            if got_echo:
                break
            logger.debug(f"  ...no answer (Attempt {attempt}/{max_tries}), trying again in {poll_interval}s")
            time.sleep(poll_interval)

        if not got_echo:
            logger.error("No echo received from board after multiple attempts. Connect handshake failed. Try to plug out and back in the board, or check the serial connection.")
            return False

        handshake_steps = [b's', b'a', b'e', b'1', b'5']
        for step in handshake_steps:
            self.send_bytes_raw(step)
            time.sleep(0.05)
            resp = self.ser.read(8)
            if resp:
                logger.debug(f"[RX  ] hex={hexdump(resp)}  ascii='{ascii_repr(resp)}'")

        logger.debug("Waiting for Boot Banner + 'Identifying device...' Prompt ...")
        banner = self.read_raw_until_idle(idle_gap=0.2, overall_timeout=5.0)

        if b"Identifying device" in banner:
            ack = banner[-4:]
            header, chunks = _split_id_blob(ID_BLOB)
            logger.debug(f"Board is waiting for identification. Sebdibg ACK ({hexdump(ack)}) "
                  f"+ G4M-Designer-Identificationblock ({len(ID_BLOB)} Bytes, "
                  f"{len(chunks)} Chunk(s) nach dem Header) ...")
            self.send_bytes_raw(bytes(ack))


            post_id = bytearray()
            self.send_bytes_raw(header)
            post_id += self.read_raw_until_idle(idle_gap=0.3, overall_timeout=2.0)
            for length, chunk in chunks:
                self.send_bytes_raw(length.to_bytes(4, "little") + chunk)
                post_id += self.read_raw_until_idle(idle_gap=0.3, overall_timeout=2.0)


            post_id += self.read_raw_until_idle(idle_gap=1.0, overall_timeout=8.0)
            post_id = bytes(post_id)
            self._report_post_id_progress(post_id)

            reached_end = b"End DirectC Demo" in post_id
            if reached_end:
                logger.info(f"'End DirectC Demo' seen -> waiting for"
                      f"{self.settle_delay:.1f}s (Settle-Delay).")
                time.sleep(self.settle_delay)
                self.power_ready = True
            else:
                logger.warning("'End DirectC Demo' not seen - the board "
                      "seems to have gotten stuck in the Boot-/DirectC-chain "
                      "(e.g., during SPI programming/verification of the FPGA-"
                      "Bitstream-Check). Power requests will likely remain "
                      "unanswered, even if the connect handshake itself appeared "
                      "normal.")
        else:
            logger.info("No 'Identifying device...' prompt seen - possibly not needed "
                         "or the board is in a different state.")

        logger.info("Handshake/Identification  done.")
        self._state = STATE_PIC
        return True

    def _report_post_id_progress(self, data: bytes):
        logger.debug("--- Post-ID Boot-/DirectC-Progress ---")
        for milestone in POST_ID_MILESTONES:
            seen = milestone in data
            logger.debug(f"  [{'x' if seen else ' '}] {milestone.decode('ascii', errors='replace')}")
        logger.debug("------------------------------------------")

    def read_raw_until_idle(self, idle_gap: float = 0.15, overall_timeout: float = 5.0) -> bytes:
        """Reads bytes untl no new bytes are received for `idle_gap` seconds or until `overall_timeout` is reached."""
        buf = bytearray()
        start = time.time()
        last = start
        while time.time() - start < overall_timeout:
            chunk = self.ser.read(1)
            if chunk:
                buf += chunk
                last = time.time()
            elif buf and (time.time() - last) > idle_gap:
                break
        if buf:
            logger.debug(f"[RX  ] hex={hexdump(bytes(buf))}")
            logger.debug(f"       ascii='{ascii_repr(bytes(buf))}'")
        return bytes(buf)

    def read_raw_next_byte(self) -> int | None:
        """Reads the next byte from the serial port, or returns None if no byte is available."""
        byte = self.ser.read(1)
        if byte:
            logger.debug(f"[RX  ] hex={hexdump(byte)}  ascii='{ascii_repr(byte)}'")
            return byte[0]
        return None

    def flush_raw(self):
        if self._raw_buf:
            data = bytes(self._raw_buf)
            self._raw_buf.clear()
            logger.debug(f"[RX  ] hex={hexdump(data)}")
            logger.debug(f"       ascii='{ascii_repr(data)}'")

    def decode_power_from_frame(self, frame: bytes):
        raw16 = (frame[1] << 8) | frame[2]
        est_watt = POWER_SCALE * raw16
        return est_watt

    def _switch_to_pic(self, NO_DELAY=False):
        if self._state == STATE_PIC:
            return True
        if self._state == STATE_DISCONNECTED:
            logger.error("Cannot switch to PIC - board is disconnected. Use connect() first.")
            return False
        if self._state == STATE_M3:
            # read all pending bytes of the M3 queue before switching to PIC
            while True:
                byte = self.read_raw_next_byte()
                if byte is None:
                    break
                self.m3_rx_queue.append(byte)
            logger.debug("Switching from M3 to PIC ...")
            self.send_bytes_raw(bytes(SWITCH_PIC_CMD))
            if not NO_DELAY:
                time.sleep(SAFETY_SWITCH_DELAY*2)
            self._state = STATE_PIC
            logger.debug("Switched to PIC.")
            return True

    def _switch_to_m3(self):
        if self._state == STATE_M3:
            return True
        if self._state == STATE_DISCONNECTED:
            logger.error("Cannot switch to M3 - board is disconnected. Use connect() first.")
            return False
        if self._state == STATE_PIC:
            logger.debug("Switching from PIC to M3 ...")
            self.send_bytes_raw(bytes([SWITCH_M3_CMD]))
            # read until the SWITCH_M3_CMD is echoed back, or until a timeout occurs
            start_time = time.time()
            while True:
                byte = self.read_raw_next_byte()
                if byte is None:
                    if time.time() - start_time > 1.0:
                        logger.error("Timeout waiting for M3 echo after switching to M3.")
                        return False
                    continue
                if byte == SWITCH_M3_CMD:
                    break
            time.sleep(SAFETY_SWITCH_DELAY)
            self._state = STATE_M3
            logger.debug("Switched to M3.")
            return True

    def get_current_power(self) -> float | None:
        """ Swicthes Mode if necessary and requests a power frame, waits and returns the estimated power in Watt. Returns None if no valid frame was received. """
        if self._state != STATE_PIC:
            if not self._switch_to_pic():
                return None
            
        self._request_power_raw()
        while True:
            frame = self.ser.read(POWER_FRAME_LEN)
            if not frame:
                logger.warning("No power frame received (timeout).")
                return None
            if len(frame) != POWER_FRAME_LEN:
                logger.warning(f"Received incomplete power frame (len={len(frame)}).")
                continue
            if frame[0] != POWER_REQUEST_BYTE:
                logger.warning(f"Received unexpected power frame (first byte={frame[0]:02X}).")
                continue
            est_watt = self.decode_power_from_frame(frame)
            logger.info(f"Received power frame: {est_watt:.3f} W")
            return est_watt

    def send_bytes_to_m3(self, data: bytes):
        """ Sends multiple bytes to the M3 core. Switches to M3 mode if necessary. """
        if self._state != STATE_M3:
            if not self._switch_to_m3():
                return
        self.send_bytes_raw(data)

    def send_text_to_m3(self, text: str):
        """ Sends a text string to the M3 core. Switches to M3 mode if necessary. """
        self.send_bytes_to_m3(text.encode("ascii", errors="replace"))

    def read_next_byte_from_m3(self) -> int | None:
        """ Reads the next byte from the M3 core. Switches to M3 mode if necessary. Returns None if no byte is available. """
        if len(self.m3_rx_queue) > 0:
            return self.m3_rx_queue.popleft()
        if self._state != STATE_M3:
            if not self._switch_to_m3():
                return None
        return self.read_raw_next_byte()

    def read_all_from_m3(self, idle_gap: float = 0.15, overall_timeout: float = 5.0) -> bytes:
        """ Reads all bytes from the M3 core until no new bytes are received for `idle_gap` seconds or until `overall_timeout` is reached. Switches to M3 mode if necessary. """
        if self._state != STATE_M3:
            if not self._switch_to_m3():
                return b""
        return self.read_raw_until_idle(idle_gap=idle_gap, overall_timeout=overall_timeout)



if __name__ == "__main__":
    # Example usage of HelloFpgaDriver
    print("Hello FPGA Driver Test Interface: ")
    p = find_mcp2221_port()
    if p is None:
        logger.error("No MCP2221 device found.")
        exit(1)
    driver = FpgaLink(port=p)
    print("Found MCP2221 device on port: ", p, " Connecting...")
    driver.connect()
    print("")
    print("Connected to device. Commands: ")
    print("p - request the current power from the PIC32")
    print("t <value> - sends the ASCII string <value> to the M3 Chip")
    print("h <value> - sends the HEX Values <value> to the M3 Chip")
    print("r - reads the next byte from the M3 Chip (only polled read available)")
    print("q - quit the test interface")

    while True:
        cmd = input("Enter command: ")
        if cmd == "q":
            break
        elif cmd == "p":
            power = driver.get_current_power()
            print(f"Current power: {power} mW")
        elif cmd.startswith("t "):
            value = cmd[2:]
            driver.send_bytes_to_m3(value.encode())
            print(f"Sent to M3: {value}")
        elif cmd.startswith("h "):
            value = cmd[2:]
            try:
                hex_bytes = bytes.fromhex(value)
                driver.send_bytes_to_m3(hex_bytes)
                print(f"Sent to M3 (hex): {hex_bytes}")
            except ValueError:
                print("Invalid hex string.")
        elif cmd == "r":
            byte = driver.read_next_byte_from_m3()
            if byte is not None:
                if (chr(byte) < ' ' or chr(byte) > '~'):
                    print(f"Received from M3: {byte} (Non-printable ASCII, Hex: {byte:02X})")
                else:
                    print(f"Received from M3: {byte} (ASCII: {chr(byte)} Hex: {byte:02X})")
            else:
                print("No byte available from M3.")
        else:
            print("Unknown command.")    
    