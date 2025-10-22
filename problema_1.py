import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leemos imagen 
img = cv2.imread('imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE)

# Verificamos propiedades de la imagen
print(type(img))
print(img.dtype)
print(img.shape)
h,w = img.shape
print("Alto: ", h)
print("Ancho: ", w)

# Función para ecualizar la imagen por bloques (local)
def equalizer (img, m, n):

    if img is None:
        print("Error: No se pudo cargar la imagen ")
        return None
    h,w = img.shape
    borde_vertical = m // 2
    borde_horizontal = n // 2

    # Agregar borde a la imagen original con el metodo REPLICATE
    img_con_borde = cv2.copyMakeBorder(img, borde_vertical, borde_vertical,
                                       borde_horizontal, borde_horizontal,
                                       cv2.BORDER_REPLICATE)

    # Imagen de salida vacía, para luego plasmar las ROIs
    img_salida = np.zeros((h, w), dtype=img.dtype)

    # Recorremos cada píxel de la imagen ORIGINAL + su borde
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # La posición correspondiente en la imagen con borde
            y_borde = y + borde_vertical
            x_borde = x + borde_horizontal

            #Extraigo la zona a analizar con un crop
            roi = img_con_borde[y_borde - borde_vertical : y_borde + borde_vertical + 1,
                                x_borde - borde_horizontal : x_borde + borde_horizontal + 1]
            
            roi_eq = cv2.equalizeHist(roi)

            img_salida[y, x] = roi_eq[borde_vertical, borde_horizontal]

    return img_salida

#Interface de Usuario por si quiere modificar los campos:
m = 16  # Alto del bloque
n = 16  # Ancho del bloque

# Ecualizamos la imagen por bloques de 16x16
img_eq = equalizer(img, m, n)
# Mostramos la imagen original y la ecualizada
cv2.imshow("Imagen Original", img)
cv2.imshow("Imagen Ecualizada por Bloques 16x16", img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
