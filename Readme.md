# Microchip Hello FPGA Python Comm
A Python Libary to communicate and measure power on the Hello FPGA Kit by Microchip

## Libary Core
The main file of the libary is ```hello_fpga_driver.py``` 
It contains the ```FpgaLink``` Class and the ```find_and_connect_to_fpga()``` function.


The ```find_and_connect_to_fpga()``` returns a ```FpgaLink``` Object of the currently connected Hello FPGA.
If you get your ```FpgaLink```like that, you do **NOT** have to call ```fpga_link.connect()```

>> **PLEASE REMEMBER TO UNPLUG AND REPLUG THE FPGA AFTER EVERY CONNECT - IT WON'T WORK WITHOUT A RESET**

### Communication Modes
The Hello FPGA has two communication modes:
- PIC32 Mode: Talk to the static Pic32 controller -> Mainly to used measure power
- M3 / FPGA Mode: Talk to your custom program on the M3 / FPGA via the UART Bridge

> The functions ```fpga_link._switch_to_pic()``` and ```fpga_link._switch_to_m3()``` can be used to switch this mode but are not required if you are not sending bytes raw. They are automatically called if you use the normal send / receive functions.

> Switching the Mode takes some time (~5ms)!

### Sending / Receiving
#### Raw Send / Receive
To send bytes / text in the exact mode you are currently in use:
- ```send_bytes_raw(bytes)```
- ```send_text_raw(text)```
- ```read_raw_next_byte()```   -> Non Blocking, if no data is there returns None 
- ```read_raw_until_idle(idle_gap overall_timeout)```

#### Send to m3 / Receive from m3
To send or receive a message to from m3, no matter which mode you are in use:
- ```send_bytes_to_m3(bytes)```
- ```send_text_to_m3(bytes)```
- ```read_next_byte_from_m3()``` -> Non Blocking, if no data is there returns None 
- ```read_all_from_m3(idle_gap overall_timeout)```
- ```await_ack(expected_ack, silently, timeout)``` -> Waits Blocking for a specific char from the m3
- ```await_ack_with_power_measure(expected_ack, silently, timeout)``` -> Waits Blocking for a specific char from the m3 and measures power while waiting (returns list of measurements)

### Power Measurements
To read the current power from the pic32, no matter which mode you are in use:
- ```get_current_power()``` -> Returns the current power usage in Watts
 


## CNN Com
The example libary ```cnn_com_functions.py``` uses the ```FpgaLink``` and implements useful helper functions to test our CNN Accelerator

- ```search_and_connect_to_fpga_and_init_m3()``` -> Searches and connects to the FPGA and goes trough init sequence with the m3, returns ```FpgaLink```
- ```send_image_array(fpga_link, image_array)``` -> Sends an 2d image array of 8bit ints to the m3
- ```send_weight_stream(fpga_link, weight_stream)``` -> Sends the 32bit weight stream (all convolutional records) to the m3 
- ```start_run_n_times(fpga_link, n, await_started_ack)``` -> Sends a start signal to the m3 so the convolution starts n times 

```python
run_power_measurement_for_image(
    fpga_link,
    image_array,
    runs_per_inference,
    power_measurements_per_inference,
    oversampling_size,
    oversampling_delay_range,
    before_measurement_delay
)
```
-> Does a full power measurement for an image.
```runs_per_infernce```: The n -> send to the start_run_n_times function gets. The number of back to back runs.
```power_measurements_per_inference```: The amount of power measurements that are done for one entire inference
```oversampling_size```: The amount of measurements repeated for one image (set to 1 if you don't want to oversample)
```oversampling_delay_range```: The range in which the wait for the oversampling is done (set to 0 is you don't want to oversample)
```before_measurement_delay```: Delay before starting the measurement -> Let the FPGA Power get 