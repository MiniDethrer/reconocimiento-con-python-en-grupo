import cv2
import os
import urllib.request # Biblioteca estándar de Python para manejar URLs

# --- Descarga del modelo Haar Cascade (usando urllib) ---
ruta_cascade = "haarcascade_frontalface_default.xml"
url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Verificar si el archivo ya existe, si no, descargarlo
if not os.path.exists(ruta_cascade):
    print("El modelo Haar Cascade no se encuentra, descargando...")
    try:
        # Usamos urlretrieve para descargar el archivo desde la URL y guardarlo en la ruta especificada
        urllib.request.urlretrieve(url_cascade, ruta_cascade)
        print("Modelo descargado exitosamente.")
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
        exit() # Salir del script si no se puede descargar el modelo

# --- Código original ---

# Cargar clasificador
cara_cascade = cv2.CascadeClassifier(ruta_cascade)

# Verificar si el clasificador se cargó correctamente
if cara_cascade.empty():
    print(f"Error al cargar el archivo cascade desde la ruta: {ruta_cascade}")
    exit()

# Iniciar captura de video
cam = cv2.VideoCapture(0)

print("Iniciando captura de video... Presiona 'q' para salir.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo obtener el frame de la cámara.")
        break

    # Convertir a gris
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    caras = cara_cascade.detectMultiScale(gris, 1.1, 4)

    # Dibujar rectángulos
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Cambié el color a verde para diferenciar

    # Mostrar el frame
    cv2.imshow("Webcam", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()