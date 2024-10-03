import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Ruta a la carpeta con las imágenes conocidas
known_faces_dir = r"C:\Users\masa\Desktop\pyton ejercicios\ia2\asisten\known_faces_dir"

# Verificar si la carpeta existe
if not os.path.exists(known_faces_dir):
    print(f"La carpeta '{known_faces_dir}' no existe. Por favor, crea esta carpeta y añade imágenes.")
    exit()

# Inicializar listas para las codificaciones y nombres conocidos
known_face_encodings = []
known_face_names = []

# Cargar imágenes conocidas
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Cargar la imagen y obtener la codificación del rostro
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:  # Verificar si se encontró una codificación
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # El nombre del archivo sin la extensión

if not known_face_encodings:
    print("No se encontraron imágenes con rostros en la carpeta conocida.")
    exit()

# Lista de índices de cámaras (0, 1, 2, etc. según el número de cámaras conectadas)
camera_indices = [0, 1]  # Añade o quita índices según el número de cámaras

# Inicializar las capturas de video para todas las cámaras
video_captures = [cv2.VideoCapture(index) for index in camera_indices]

# Inicializar estructura de datos para registrar asistencia
attendance_data = {}

while True:
    for capture in video_captures:
        # Capturar un fotograma de la cámara
        ret, frame = capture.read()

        # Verificar si la cámara está capturando
        if not ret:
            continue

        # Convertir el fotograma a RGB (ya que face_recognition trabaja con imágenes en RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encontrar las ubicaciones de los rostros en el fotograma
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Iterar sobre los rostros detectados en el fotograma
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparar el rostro detectado con los rostros conocidos
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"  # Nombre predeterminado si la cara no es reconocida

            # Usar la distancia más corta para encontrar el mejor match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Extraer el grupo del nombre
            if '_' in name:
                group = name.split('_')[-1]
            else:
                group = "No_Grupo"

            # Registrar asistencia
            if group not in attendance_data:
                attendance_data[group] = pd.DataFrame(columns=["Nombre", "Fecha", "Hora"])
            if name != "Desconocido" and name not in attendance_data[group]["Nombre"].values:
                now = datetime.now()
                new_entry = pd.DataFrame([{"Nombre": name, "Fecha": now.date(), "Hora": now.time()}])
                attendance_data[group] = pd.concat([attendance_data[group], new_entry], ignore_index=True)

            # Extraer las coordenadas del rostro
            top, right, bottom, left = face_location

            # Dibujar un rectángulo alrededor del rostro detectado
            color = (0, 0, 255) if name == "Desconocido" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Escribir el nombre sobre el rectángulo
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

        # Mostrar el fotograma con los rostros detectados y reconocidos
        cv2.imshow(f"Face Recognition - Camera {video_captures.index(capture)}", frame)

    # Si se presiona 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Exportar la asistencia a un archivo de Excel por grupo
output_dir = r"C:\Users\masa\Desktop\pyton ejercicios\ia2\asisten"
for group_name, attendance_df in attendance_data.items():
    file_name = f"{group_name}_asistencia.xlsx"
    file_path = os.path.join(output_dir, file_name)
    attendance_df.to_excel(file_path, index=False)
    print(f"Asistencia para el grupo {group_name} exportada a '{file_path}'")

# Liberar las capturas de video y cerrar las ventanas
for capture in video_captures:
    capture.release()
cv2.destroyAllWindows()
