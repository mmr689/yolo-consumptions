## Using GPIO on Rock4C+ with the `periphery` library

### Introduction

This guide demonstrates how to control General Purpose Input/Output (GPIO) pins on the Rock4C+ using the `periphery` Python library.

### Prerequisites

Ensure you have Python installed on your system. This example uses Python 3.9.2 and the `periphery` installed version is `python-periphery==2.4.1`.

### Installation

First, you need to install the `periphery` library. You can install it using `pip` by running the following command in your terminal:

```bash
pip install python-periphery
```

### Example Code

Here's a simple example to show how to configure and control a GPIO pin:

```python
from periphery import GPIO
import time

# Configure the GPIO for output
gpio = GPIO("/dev/gpiochip3", 15, "out") # Assuming we want to use chip 3 and GPIO 15.

try:
    # Set GPIO high (1)
    gpio.write(True)
    print("GPIO set high")
    time.sleep(2)  # Keep high for 2 seconds

    # Set GPIO low (0)
    gpio.write(False)
    print("GPIO set low")

finally:
    # Ensure to clean up and close the GPIO
    gpio.close()
```

### Checking GPIO Status

To check the status of GPIOs, you can use the `gpioinfo` command from the console. To select a pin that you can control for generating rising and falling edges, look for one that is not in use (`unused`) and can be configured as an output (`output`).