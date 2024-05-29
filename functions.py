import os

def check_and_create_folder(folder_path):
    """
    Checks if a folder exists at the specified path. If it doesn't exist,
    creates the folder.

    Parameters:
    folder_path (str): The path of the folder to check and create if it doesn't exist.

    Returns:
    None
    """
    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
        print(f'Folder {folder_path} has been created')



import pysftp

def get_data(host_ip, host_username, host_password):
    """
    Función para conectarnos al NAS
    """
    try:
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        scp = pysftp.Connection(host_ip, username=host_username, password=host_password, cnopts=cnopts)
    except pysftp.AuthenticationException as auth_error:
        print(f'Error de autenticación: {auth_error}')
        scp = None
    except Exception as e:
        print(f'Otro error: {e}')
        scp = None
    else:
        print('Conexión exitosa')
        return scp
    
def descargar_archivos_conteniendo_string(scp, directorio_remoto, string_a_buscar, directorio_local):
    """
    Función para descargar un archivo filtrado por nombre
    """
    try:
        # Cambia al directorio remoto
        scp.chdir(directorio_remoto)
        # Imprime la ubicación actual
        # print(f'Estás en el directorio remoto: {scp.pwd}')

        # Enumera todos los archivos en el directorio remoto
        lista_archivos = scp.listdir()
        # print(f'Los archivos presentes en el directorio remoto son: {lista_archivos}')

        # Filtra los archivos que contienen el string en su nombre y son archivos .jpg
        imagenes = [archivo for archivo in lista_archivos if string_a_buscar in archivo and archivo.lower().endswith('.jpg')]
        # print(f'Las imágenes son: {imagenes}')
        # Descarga las imágenes coincidentes al directorio local
        for imagen in imagenes:
            get_file(scp, remote_path= '', local_path= directorio_local, name=imagen)

        print(f'Se han descargado {len(imagenes)} archivos coincidentes.')

    except Exception as e:
        print(f'Ocurrió un error al descargar archivos: {e}')

def get_file(scp, remote_path, local_path, name):
    """
    Este creo que es para un documento que conocemos exactamente el nombre
    """
    if scp is not None:
        try:
            scp.get(remote_path+name, local_path+name)
        except:
            print('Error GET')
        else:
            print('Descarga exitosa', remote_path+name)