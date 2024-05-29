"""
Código para mover carpetas dentro del servidor
"""

import shutil
import os

# Definir las rutas absolutas
ruta_origen = '/home/qcienmed/mmr689/yolo-comp2/datasets/bioview'
ruta_destino = '/home/qcienmed/mmr689/yolo-consumptions/datasets'

# Verificar si el destino es un directorio existente
if os.path.isdir(ruta_destino):
    # Construir la ruta completa del destino
    destino_completo = os.path.join(ruta_destino, os.path.basename(ruta_origen))
else:
    destino_completo = ruta_destino

# Mover la carpeta
try:
    shutil.move(ruta_origen, destino_completo)
    print(f"Carpeta movida de {ruta_origen} a {destino_completo} exitosamente.")
except Exception as e:
    print(f"Ocurrió un error al mover la carpeta: {e}")