En el siguiente Repositorio "TP_PDI_2025" se presentan los puntos 1) y 2) del Trabajo Practico de Procesamiento de Imagenes.

Para ejecutar el mismo, debe seguir los siguientes pasos:
  1) Clonar el repositorio con el siguiente comando en Consola --> gh repo clone jgottig/TP_PDI_2025
  2) Previamente, debe tener instaladas las librerías necesarias con los siguientes comandos:
      pip install opencv-python
      pip install numpy
      pip install matplotlib
      Ademas, utilizamos librerías nativas de Python
  3) Para su ejecución, en algunos casos recomendamos que utilice la versión de Python 3.10 (Dado que OpenCV puede tener fallas en Versiones 3.11 +)
  4) Abrir su IDE Favorito.
     
Ejecución Problema 1)
  1) Guardar la imagen a analizar bajo el nombre "Imagen_con_detalles_escondidos.tif" en la misma carpeta donde se encuentra el documento "problema_1.py" (Tal como está ahora)
  2) Ejecutar el archivo "problema_1.py" completo.
  3) En caso que quiera validar distintas opciones varíando el tamaño del bloque, puede modificar los campos
      m = 16  # Alto del bloque
      n = 16  # Ancho del bloque
  4) Comparar entre la imagen original y la imagen ecualizada localmente.

Ejecución Problema 2)
  1) Dentro de la carpeta Formularios debe guardar los formularios a corregir en formato .png
  2) Cada formulario, debe renombrarlo como "formulario_"id".png, ejemplo "formulario_07.png"
  3) El archivo "problema_2.py" debe estar en la misma carpeta que la carpeta Formularios (Tal como está ahora)
  4) Ejecute el archivo "problema_2.py" completo.
  5) Analice los resultados en la imagen de salida para validar quienes completaron el formulario OK y quienes Mal
  6) Analice las respuestas en el CSV de Salida para comprender cuales fueron esas respuestas erroneas.
