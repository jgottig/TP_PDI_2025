import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re

def obtener_imagenes_formularios():

    # Sus archivos deben estar dentro de la carpeta Formularios
    nombre_carpeta = 'Formularios/'
    lista_imagenes = {}

    if not os.path.isdir(nombre_carpeta):
        print(f"Error: La carpeta '{nombre_carpeta}' no se encontró.")
        return lista_imagenes

    # Recorrer todos los archivos en el directorio 'Formularios'
    for nombre_archivo in os.listdir(nombre_carpeta):
        ruta_completa = os.path.join(nombre_carpeta, nombre_archivo)
            
        # Leer la imagen en escala de grises
        img = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
            
        # Si la imagen se leyó correctamente, añadirla a la lista
        if img is not None:
            lista_imagenes[nombre_archivo] = img
        else:
            print(f"Advertencia: No se pudo leer la imagen '{nombre_archivo}'.")

    return lista_imagenes

#Utilizamos la función para detectar las posiciones de las líneas
def get_line_positions(line_detection_array):
    positions = []
    is_line = False
    start = 0
    for i, val in enumerate(line_detection_array):
        if val and not is_line:
            is_line = True
            start = i
        elif not val and is_line:
            is_line = False
            # Se toma el punto medio de la línea detectada
            positions.append(start + (i - 1 - start) // 2)
    
    return positions


#En esta función, recortaremos los formularios para quedarnos unicamente
#con las celdas de interés, en la sección de respuestas.
def recortar_formulario(img):

    renglones_recortados = []
    th_value = 200
    img_th = (img < th_value).astype(np.uint8) * 255  #Saturamos la imagen, para que quede binaria
    img_cols = np.sum(img_th, axis=0) / 255
    img_rows = np.sum(img_th, axis=1) / 255
    
    #Establecemos umbrales para detectar líneas, uno para Columnas y otro para Filas.
    th_col = img_cols.max()*0.90 
    th_row = img_rows.max()*0.95

    rows_with_lines = img_rows > th_row
    cols_with_lines = img_cols > th_col

    horizontal_lines = get_line_positions(rows_with_lines)
    vertical_lines = get_line_positions(cols_with_lines)

    # Bucle para procesar cada renglón delimitado por las líneas horizontales
    for i in range(len(horizontal_lines) - 1):
        y1 = horizontal_lines[i]
        y2 = horizontal_lines[i+1]

        #Conocemos que nuestras preguntas se encuentran en los renglones 6/7/8 (índices 6,7,8)
        indices_preguntas = [6, 7, 8] # Ajusta estos índices si tu formulario cambia

        if i in indices_preguntas:
            #Recortar el área que contiene las respuestas "Si" y "No"
            x_inicio_respuestas = vertical_lines[1] + 15 # Columna izquierda
            x_fin_respuestas = vertical_lines[-1] - 15   # Borde derecho
            
            #Hacemos pequeños ajustes para evitar líneas
            area_respuestas = img[y1+3:y2-3, x_inicio_respuestas:x_fin_respuestas]

            # PASO 2: Dentro de la sub-area, buscamos su linea vertical divisoria
            area_th = (area_respuestas < th_value).astype(np.uint8) * 255
            suma_vertical = np.sum(area_th, axis=0) / 255
            
            ancho_area = area_respuestas.shape[1]
            
            # Buscamos la línea en el tercio central para evitar ruido en los bordes
            inicio_busqueda = ancho_area // 3
            fin_busqueda = 2 * ancho_area // 3
            
            # Si hay un pico suficientemente alto (una línea), la usamos para el corte
            if np.max(suma_vertical[inicio_busqueda:fin_busqueda]) > (area_respuestas.shape[0] * 0.5): # Si la línea ocupa más del 50% de la altura
                pos_corte_relativa = np.argmax(suma_vertical[inicio_busqueda:fin_busqueda])
                pos_corte_absoluta = inicio_busqueda + pos_corte_relativa
                
                celda_si = area_respuestas[:, :pos_corte_absoluta]
                celda_no = area_respuestas[:, pos_corte_absoluta+3:] # +1 para saltar la línea

            renglones_recortados.append(celda_si)
            renglones_recortados.append(celda_no)

        else:
            # Para los renglones que no son de preguntas
            # Se extrae solo la celda de contenido, no la etiqueta.
            x1_contenido = vertical_lines[1] + 5
            x2_contenido = vertical_lines[-1] - 7
            renglon = img[y1+3:y2-3, x1_contenido:x2_contenido]
            renglones_recortados.append(renglon)

    return renglones_recortados

#Analisis de los renglones recortados
def analizar_renglones(renglones):

    lista_de_resultados = []
    for idx, renglon_img in enumerate(renglones):

        #Saltar los renglones con índice 0 y 5 por corresponderse a Títutlos
        if idx == 0 or idx == 5:
            continue
        
        #Binarizamos la sub-imagen del renglón
        umbral, celda_binaria = cv2.threshold(renglon_img, 200, 255, cv2.THRESH_BINARY_INV)
    
        #Buscamos sus componentes 8-conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_binaria, 8, cv2.CV_32S)
        
        #Aplicamos un umbral de 15px de area para evitar el ruido
        th_area = 15
        stats_filtrados = stats[1:][stats[1:, cv2.CC_STAT_AREA] > th_area]
        num_caracteres = len(stats_filtrados)

        if num_caracteres == 0:
            num_palabras = 0
        else:
            #Metodología para contar palabras basada en espacios entre caracteres
            #    Ordenamos los caracteres de izquierda a derecha
            #    MEdida para encontrar Espacio - distancia mayor a un porcentaje del ancho promedio de un caracter
            #    Recorremos los caracteres para encontrar espacios grandes entre ellos
            stats_ordenados = stats_filtrados[stats_filtrados[:, cv2.CC_STAT_LEFT].argsort()]
            ancho_promedio_char = np.mean(stats_ordenados[:, cv2.CC_STAT_WIDTH])
            umbral_espacio = ancho_promedio_char * 0.7 

            num_palabras = 1
            for i in range(1, num_caracteres):
                fin_anterior = stats_ordenados[i-1, cv2.CC_STAT_LEFT] + stats_ordenados[i-1, cv2.CC_STAT_WIDTH]
                inicio_actual = stats_ordenados[i, cv2.CC_STAT_LEFT]
                
                # Si la distancia es mayor que nuestro umbral, contamos una nueva palabra
                if (inicio_actual - fin_anterior) > umbral_espacio:
                    num_palabras += 1
                    
        # Imprimir el resultado para el renglón actual
        lista_de_resultados.append((idx, num_caracteres, num_palabras, renglon_img))
    return lista_de_resultados

def principal():
    #Creamos variables que utilizaremos para armar la imagen de salida y los resultados
    imagen_salida = np.zeros((1000, 1000), dtype=np.uint8)  # Creamos imagen negra grande para la salida
    posicion_y_actual = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala_font = 1.2
    color_texto = (255, 255, 255)
    grosor_texto = 2
    resultados = {}

    #Obtenemos las imágenes de los formularios y los analizamos uno por uno
    formularios = obtener_imagenes_formularios()

    for nombre_formulario, img_formulario in formularios.items():

        resultado_nombre_y_apellido = ""
        resultado_edad = ""
        resultado_mail = ""
        resultado_legajo = ""
        resultado_pregunta_1 = ""
        resultado_pregunta_2 = ""
        resultado_pregunta_3 = ""
        resultado_comentarios = ""        
        renglones = recortar_formulario(img_formulario)

        #Reglas y restricciones para analizar el cumplimiento de cada renglon según consignas
        for idx, num_caracteres, num_palabras, imagen in analizar_renglones(renglones):
            #Nombre y Apellido
            if idx == 1:
                if num_palabras >= 2 and num_caracteres < 25:
                    resultado_nombre_y_apellido = "OK"    
                else:
                    resultado_nombre_y_apellido = "MAL"
                imagen_nombre_apellido = imagen
                
            #Edad
            if idx == 2:
                if 2 <= num_caracteres <= 3 and num_palabras == 1:
                    resultado_edad = "OK"
                else:
                    resultado_edad = "MAL"

            #Mail
            if idx == 3:
                if num_palabras == 1 and num_caracteres < 25:
                    resultado_mail = "OK"
                else:
                    resultado_mail = "MAL"
            
            #Legajo
            if idx == 4:
                if num_caracteres == 8 and num_palabras == 1:
                    resultado_legajo = "OK"
                else:
                    resultado_legajo = "MAL"

            #Analisis pregunta 1        
            if idx == 6:  # Celda SI
                info_celda_si = {'caracteres': num_caracteres, 'palabras': num_palabras}

            if idx == 7:  # Celda NO
                # Extraemos la info de la celda SI que guardamos antes
                num_caracteres_si = info_celda_si['caracteres']
                num_palabras_si = info_celda_si['palabras']
                print(f"Pregunta 1 - Celda SI: {num_caracteres_si} caracteres, {num_palabras_si} palabras.")

                # Usamos la info actual para la celda NO
                num_caracteres_no = num_caracteres
                num_palabras_no = num_palabras
                print(f"Pregunta 1 - Celda NO: {num_caracteres_no} caracteres, {num_palabras_no} palabras.")

                # Aplicamos la validación
                condicion1_ok = (num_caracteres_si == 1 and num_palabras_si == 1) and \
                                (num_caracteres_no == 0 and num_palabras_no == 0)

                condicion2_ok = (num_caracteres_si == 0 and num_palabras_si == 0) and \
                                (num_caracteres_no == 1 and num_palabras_no == 1)

                if condicion1_ok or condicion2_ok:
                    resultado_pregunta_1 = "OK"
                else:
                    resultado_pregunta_1 = "MAL"

            #Analisis pregunta 2 - Misma metodología que pregunta 1       
            if idx == 8: # Celda SI
                info_celda_si = {'caracteres': num_caracteres, 'palabras': num_palabras}

            if idx == 9:  # Celda NO
                num_caracteres_si = info_celda_si['caracteres']
                num_palabras_si = info_celda_si['palabras']
                num_caracteres_no = num_caracteres
                num_palabras_no = num_palabras

                condicion1_ok = (num_caracteres_si == 1 and num_palabras_si == 1) and \
                                (num_caracteres_no == 0 and num_palabras_no == 0)

                condicion2_ok = (num_caracteres_si == 0 and num_palabras_si == 0) and \
                                (num_caracteres_no == 1 and num_palabras_no == 1)

                if condicion1_ok or condicion2_ok:
                    resultado_pregunta_2 = "OK"
                else:
                    resultado_pregunta_2 = "MAL"

            #Analisis pregunta 3 - Misma metodología que pregunta 1        
            if idx == 10:  # Celda SI
                info_celda_si = {'caracteres': num_caracteres, 'palabras': num_palabras}

            if idx == 11:  # Celda NO
                num_caracteres_si = info_celda_si['caracteres']
                num_palabras_si = info_celda_si['palabras']
                
                num_caracteres_no = num_caracteres
                num_palabras_no = num_palabras

                condicion1_ok = (num_caracteres_si == 1 and num_palabras_si == 1) and \
                                (num_caracteres_no == 0 and num_palabras_no == 0)

                condicion2_ok = (num_caracteres_si == 0 and num_palabras_si == 0) and \
                                (num_caracteres_no == 1 and num_palabras_no == 1)

                if condicion1_ok or condicion2_ok:
                    resultado_pregunta_3 = "OK"
                else:
                    resultado_pregunta_3 = "MAL"

            #Analisis comentarios
            if idx == 12:
                if num_palabras >= 1 and num_caracteres < 25:
                    resultado_comentarios = "OK"
                else:
                    resultado_comentarios = "MAL"
            
        # Imprimimos los resultados como pedía el Enunciado    
        print(f"Resultados para el formulario '{nombre_formulario}':")
        print(f"Nombre y Apellido: {resultado_nombre_y_apellido}")
        print(f"Edad: {resultado_edad}")
        print(f"Mail: {resultado_mail}")
        print(f"Legajo: {resultado_legajo}")
        print(f"Pregunta 1: {resultado_pregunta_1}")
        print(f"Pregunta 2: {resultado_pregunta_2}")
        print(f"Pregunta 3: {resultado_pregunta_3}")
        print(f"Comentarios: {resultado_comentarios}")

        #Extraemos el ID del nombre del archivo usando Regex
        id = (re.findall(r'(?<=_)\d+(?=\.)', nombre_formulario) + [None])[0]
        print(id)

        #Guardamos los resultados en un diccionario para luego crear el CSV
        resultados[id] = [
            resultado_nombre_y_apellido, resultado_edad, resultado_mail,
            resultado_legajo, resultado_pregunta_1, resultado_pregunta_2,
            resultado_pregunta_3, resultado_comentarios]
        
        # Establecemos posiciones a utilizar para pegar la imagen y el texto
        x1 = 10
        x2 = x1 + imagen_nombre_apellido.shape[1]
        y1 = idx
        y2 = y1 + imagen_nombre_apellido.shape[0]

        if resultado_nombre_y_apellido == "OK" and resultado_edad == "OK" and \
            resultado_mail == "OK" and resultado_legajo == "OK" and \
            resultado_pregunta_1 == "OK" and resultado_pregunta_2 == "OK" and \
            resultado_pregunta_3 == "OK" and resultado_comentarios == "OK":
            print("El formulario está correctamente completado.\n")

            # Calculamos las coordenadas para pegar la imagen
            x1 = 10
            x2 = x1 + imagen_nombre_apellido.shape[1]
            y1 = posicion_y_actual
            y2 = y1 + imagen_nombre_apellido.shape[0]

            # Pegamos la imagen del nombre y apellido
            imagen_salida[y1:y2, x1:x2] = imagen_nombre_apellido
            texto_a_escribir = "OK"
            posicion_texto_x = x2 + 20  # Unos 20 píxeles a la derecha de donde termina la imagen

            # Para centrar verticalmente el texto con la imagen
            alto_imagen = imagen_nombre_apellido.shape[0]
            posicion_texto_y = y1 + int(alto_imagen * 0.8) # Posición base del texto
            cv2.putText(imagen_salida, texto_a_escribir, (posicion_texto_x, posicion_texto_y), font, escala_font, color_texto, grosor_texto)
            
            # Actualizamos la posición vertical para la siguiente imagen
            posicion_y_actual += imagen_nombre_apellido.shape[0] + 5 # +5 para un pequeño espacio
        else:
            print("El formulario tiene errores en el llenado.\n")    
            x1 = 10
            x2 = x1 + imagen_nombre_apellido.shape[1]
            y1 = posicion_y_actual
            y2 = y1 + imagen_nombre_apellido.shape[0]

            # Pegamos la imagen del nombre y apellido + Su Calificación
            imagen_salida[y1:y2, x1:x2] = imagen_nombre_apellido
            texto_a_escribir = "MAL"
            posicion_texto_x = x2 + 20

            # Para centrar verticalmente el texto con la imagen
            alto_imagen = imagen_nombre_apellido.shape[0]
            posicion_texto_y = y1 + int(alto_imagen * 0.8) # Posición base del texto
            cv2.putText(imagen_salida, texto_a_escribir, (posicion_texto_x, posicion_texto_y), font, escala_font, color_texto, grosor_texto)

            # Actualizamos la posición vertical para la siguiente imagen
            posicion_y_actual += imagen_nombre_apellido.shape[0] + 5 # +5 para un pequeño espacio
    
    # Mostramos la imagen resumen de todos los formularios como pedía el enunciado            
    cv2.imshow("Resumen de Formularios", imagen_salida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()        
    return resultados

#Creamos función para crear el CSV con los resultados
def crear_csv(resultados):

    nombre_archivo_csv = 'resultados_formularios.csv'
    encabezados = ['ID', 'Nombre y Apellido', 'Edad', 'Mail', 'Legajo', 'Pregunta 1', 'Pregunta 2', 'Pregunta 3', 'Comentarios']

    # Abrir el archivo en modo escritura (En caso que no existe, lo crea)
    with open(nombre_archivo_csv, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(encabezados)
        # Escribir los resultados de cada formulario
        for id, resultados_formulario in resultados.items():
            fila = [id] + resultados_formulario
            escritor_csv.writerow(fila)

# Función del usuario: --> Solo ejecutar. 
# Dar play y los resultados estarán listos 
# y guardados en un CSV llamado "resultados_formularios.csv"
# :)

crear_csv(principal())
