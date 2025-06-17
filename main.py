import cv2
import os
import urllib.request

# --- Descarga del modelo Haar Cascade para rostros (usando urllib) ---
ruta_cascade_cara = "haarcascade_frontalface_default.xml"
url_cascade_cara = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

if not os.path.exists(ruta_cascade_cara):
    print("El modelo Haar Cascade para rostros no se encuentra, descargando...")
    try:
        urllib.request.urlretrieve(url_cascade_cara, ruta_cascade_cara)
        print("Modelo de rostros descargado exitosamente.")
    except Exception as e:
        print(f"Error al descargar el modelo de rostros: {e}")
        exit()

# --- Descarga del modelo Haar Cascade para ojos (usando urllib) ---
ruta_cascade_ojo = "haarcascade_eye.xml"
url_cascade_ojo = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

if not os.path.exists(ruta_cascade_ojo):
    print("El modelo Haar Cascade para ojos no se encuentra, descargando...")
    try:
        urllib.request.urlretrieve(url_cascade_ojo, ruta_cascade_ojo)
        print("Modelo de ojos descargado exitosamente.")
    except Exception as e:
        print(f"Error al descargar el modelo de ojos: {e}")
        exit()

# --- Código original modificado para reconocimiento de ojos ---

# Cargar clasificadores
cara_cascade = cv2.CascadeClassifier(ruta_cascade_cara)
ojo_cascade = cv2.CascadeClassifier(ruta_cascade_ojo)

# Verificar si los clasificadores se cargaron correctamente
if cara_cascade.empty():
    print(f"Error al cargar el archivo cascade de rostros desde la ruta: {ruta_cascade_cara}")
    exit()
if ojo_cascade.empty():
    print(f"Error al cargar el archivo cascade de ojos desde la ruta: {ruta_cascade_ojo}")
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

    # Dibujar rectángulos en los rostros y luego detectar ojos dentro de cada rostro
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Rectángulo verde para el rostro

        # Región de interés (ROI) para el rostro en escala de grises y color
        roi_gris = gris[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar ojos dentro de la región del rostro
        ojos = ojo_cascade.detectMultiScale(roi_gris)
        for (ex, ey, ew, eh) in ojos:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2) # Rectángulo azul para los ojos

    # Mostrar el frame
    cv2.imshow("Webcam - Reconocimiento Facial y Ocular", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()
