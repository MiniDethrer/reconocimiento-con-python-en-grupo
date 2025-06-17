import cv2
import os
import urllib.request
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk # Used for displaying OpenCV images in Tkinter

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
        messagebox.showerror("Error de Descarga", f"No se pudo descargar el modelo Haar Cascade: {e}")
        exit()

# --- Cargar clasificador ---
cara_cascade = cv2.CascadeClassifier(ruta_cascade)

# Verificar si el clasificador se cargó correctamente
if cara_cascade.empty():
    messagebox.showerror("Error de Carga", f"Error al cargar el archivo cascade desde la ruta: {ruta_cascade}")
    exit()

# --- Variables globales para el control de la cámara ---
cam = None
running = False
video_stream_id = None # To hold the ID of the after() method

# --- Funciones para el GUI ---

def start_camera():
    global cam, running, video_stream_id
    if running:
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error de Cámara", "No se pudo iniciar la cámara. Asegúrate de que no esté en uso.")
        return

    running = True
    print("Iniciando captura de video...")
    update_frame() # Start updating the frames

def stop_camera():
    global cam, running, video_stream_id
    if not running:
        return

    running = False
    print("Deteniendo captura de video...")
    if cam:
        cam.release()
    if video_stream_id:
        app.after_cancel(video_stream_id) # Stop the scheduled frame updates
    
    # Clear the canvas when stopping the camera
    canvas.delete("all")
    lmain.config(image='') # Clear the image on the label
    
    cam = None # Reset camera object

def update_frame():
    global cam, running, video_stream_id
    if running and cam:
        ret, frame = cam.read()
        if ret:
            # Convertir a gris
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostros
            caras = cara_cascade.detectMultiScale(gris, 1.1, 4)

            # Dibujar rectángulos
            for (x, y, w, h) in caras:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convertir el frame de OpenCV a un formato que Tkinter pueda mostrar
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        else:
            print("No se pudo obtener el frame de la cámara.")
            stop_camera() # Stop if no frame can be read
            messagebox.showwarning("Advertencia", "No se pudo obtener el frame de la cámara. Deteniendo.")

        video_stream_id = app.after(10, update_frame) # Schedule the next frame update after 10ms
    elif not running and video_stream_id:
        app.after_cancel(video_stream_id) # Ensure the loop stops if running becomes False

# --- Configuración de la Ventana Tkinter ---
app = tk.Tk()
app.title("Detección de Rostros con Webcam")
app.geometry("800x600")

# Frame para los botones
button_frame = tk.Frame(app)
button_frame.pack(pady=10)

# Botón para iniciar la cámara
start_button = tk.Button(button_frame, text="Iniciar Cámara", command=start_camera, font=("Arial", 12))
start_button.pack(side=tk.LEFT, padx=10)

# Botón para detener la cámara
stop_button = tk.Button(button_frame, text="Detener Cámara", command=stop_camera, font=("Arial", 12))
stop_button.pack(side=tk.LEFT, padx=10)

# Etiqueta para mostrar el video (donde se cargará la imagen de la cámara)
lmain = tk.Label(app)
lmain.pack()

# Canvas (optional, if you want to draw shapes directly on it, but Label is better for images)
canvas = tk.Canvas(app, width=640, height=480, bg="black")
# canvas.pack() # You can uncomment this if you prefer canvas for drawing, but for simple image display, Label is easier.

# Manejar el cierre de la ventana
def on_closing():
    if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
        stop_camera()
        app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# Iniciar el bucle principal de la aplicación Tkinter
app.mainloop()
