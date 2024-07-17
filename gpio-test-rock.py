""" FUNCIONA"""
from periphery import GPIO
import time

# Configurar el GPIO para salida
chip = 3
gpio = 15
gpio = GPIO(f"/dev/gpiochip{chip}", gpio, "out")

try:
    # Poner el GPIO a alto (1)
    gpio.write(True)
    print("GPIO puesto a alto")
    time.sleep(2)  # Mantener alto durante 2 segundos

    # Poner el GPIO a bajo (0)
    gpio.write(False)
    print("GPIO puesto a bajo")

finally:
    # Asegúrate de limpiar y cerrar el GPIO
    gpio.close()
