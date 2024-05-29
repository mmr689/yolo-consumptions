import os
import datetime as dt

from private_data import *
from functions import check_and_create_folder, descargar_archivos_conteniendo_string, get_data

cam = '04'
pos = 'T'


host_ip = nas_host_ip # IP del servidor remoto
host_username = nas_host_username  # Nombre de usuario remoto del servidor
host_password = nas_host_password # Contraseña remota del servidor

remote_path = f'./ENTOITRAPS/BIOVIEWALORDA/BIOVIEW{cam}/CAM{pos}/'
local_path  = 'datasets/bioview-test/'



# Generamos lista de fechas en las que nos queremos descargar las fotos
start_data = dt.datetime(2022, 12, 21)
end_data = dt.datetime(2022, 12, 23)
# Crear la lista de fechas en formato YYYYMMDD usando una lista comprimida
dates = [now.strftime("%Y%m%d")
         for now in (start_data + dt.timedelta(days=d)
            for d in range((end_data - start_data).days + 1))]
# Recorremos las fechas y descargamos los datos asociados a ellas.
for date in dates:
    # ############################################## #
    # ############ NAS DATA DOWNLOAD ############### #
    # ############################################## #
    # Ruta completa de la carpeta
    folder_server_name = f'{date}_CAM{cam}/'
    folder_path = os.path.join(local_path,'imgs', folder_server_name)
    # Crear la carpeta si no existe
    check_and_create_folder(folder_path)
    # Actualizamos la ruta de trabajo
    final_path = os.path.join(local_path,'imgs', f'{date}_CAM{cam}/')
    # Nos conectamos al NAS
    scp = get_data(host_ip, host_username, host_password)
    # Nos descargamos las imagenes
    if scp:
        descargar_archivos_conteniendo_string(scp, remote_path, date, final_path)
        scp.close()