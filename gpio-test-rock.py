""" FUNCIONA"""
from periphery import GPIO
import time

# Configurar el GPIO para salida
# Reemplaza 'X' con el número de 'chip' y 'Y' con el número de línea basado en tu selección
gpio = GPIO("/dev/gpiochip3", 15, "out")

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
