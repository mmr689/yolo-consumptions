# Visualizador de Imágenes con Anotaciones

Este módulo de Python permite cargar y visualizar imágenes con anotaciones, como rectángulos y segmentaciones, utilizando únicamente Matplotlib. Es ideal para visualizar datos de imágenes con anotaciones simples sin necesidad de dependencias adicionales.

## Características

- **Carga de imágenes**: Utiliza Matplotlib para cargar y mostrar imágenes.
- **Anotación de imágenes**: Permite dibujar rectángulos y segmentaciones sobre la imagen.
- **Personalización**: Soporta personalización del tamaño de la figura y títulos.

## Requisitos

Este código requiere Python con la biblioteca Matplotlib instalada. Asegúrate de tener las siguientes versiones instaladas para evitar problemas de compatibilidad:

- Python 3.8.10
- Matplotlib 3.7.5

Puedes instalar Matplotlib usando pip si aún no está instalado:

```bash
pip install matplotlib
```

## Uso

Para utilizar este módulo, simplemente importa la función `plot_image_with_annotations` en tu script de Python y llama a la función con los parámetros necesarios.

### Parámetros de la Función

- `path` (str): El camino hacia el directorio donde se encuentra la imagen.
- `filename` (str): El nombre del archivo de la imagen.
- `rectangles` (list of tuples, opcional): Lista de tuplas (x, y, w, h) para dibujar rectángulos en la imagen.
- `segmentations` (list of list of floats, opcional): Lista que contiene puntos de contorno intercalados [x1, y1, x2, y2, ...].
- `figsize` (tuple, opcional): Tamaño de la figura.
- `title` (str, opcional): Título de la imagen.

### Formato de las Coordenadas

- **Rectángulos**: Cada rectángulo se define por una tupla `(x, y, w, h)`, donde `x, y` son las coordenadas del punto superior izquierdo del rectángulo, y `w, h` son el ancho y alto del rectángulo respectivamente.
- **Segmentaciones**: Cada segmentación se define por una lista de coordenadas intercaladas, donde cada par consecutivo de números representa las coordenadas `x, y` de un punto en el contorno del área segmentada.

### Ejemplo de Uso

```python
plot_image_with_annotations(
    '/path/to/images', 'image.jpg',
    rectangles=[(50, 50, 100, 100)],
    segmentations=[[247.71, 354.7, 253.49, 346.99, 276.63, 337.35, 312.29, 333.49]],
    figsize=(12, 10), title='Imagen con Anotaciones'
)
```

Este ejemplo cargará la imagen `image.jpg` del directorio `/path/to/images`, dibujará un rectángulo y una segmentación, y mostrará la imagen con una figura de tamaño 12x10 pulgadas y un título.