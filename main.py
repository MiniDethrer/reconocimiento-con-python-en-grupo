import cv2
import os
import urllib.request

# --- Descarga del modelo Haar Cascade (usando urllib) ---
ruta_cascade = "haarcascade_frontalface_default.xml"
url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Verificar si el archivo ya existe, si no, descargarlo
if not os.path.exists(ruta_cascade):
    print("El modelo Haar Cascade no se encuentra, descargando...")
    try:
        urllib.request.urlretrieve(url_cascade, ruta_cascade)
        print("Modelo descargado exitosamente.")
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
        exit()

# --- Código de preprocesamiento ---

# Cargar clasificador
cara_cascade = cv2.CascadeClassifier(ruta_cascade)

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

    # Aplicar el filtro de Canny
    bordes = cv2.Canny(gris, 100, 200)

    # Detectar rostros en la imagen original
    caras = cara_cascade.detectMultiScale(gris, 1.1, 4)

    # Dibujar rectángulos para las caras detectadas
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el frame original y el de bordes
    cv2.imshow("Webcam - Original", frame)
    cv2.imshow("Webcam - Bordes", bordes)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()
