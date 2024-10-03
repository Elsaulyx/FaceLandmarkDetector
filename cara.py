import cv2
import mediapipe as mp

# Función para obtener la posición relativa de los puntos de la cara
def obtener_posicion_cara(face_landmarks, frame_shape):
    puntos_cara = []
    altura, ancho, _ = frame_shape
    
    for punto, landmark in enumerate(face_landmarks.landmark):
        x, y = int(landmark.x * ancho), int(landmark.y * altura)
        puntos_cara.append((x, y, punto))  # Agregar el número del landmark
    
    return puntos_cara

# Función para dibujar el texto en la imagen
def dibujar_texto(frame, texto, posicion, color_texto):
    cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)

# Función para dibujar los números de los landmarks
def dibujar_numeros(frame, puntos_cara):
    for (x, y, num) in puntos_cara:
        cv2.putText(frame, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

# Función para dibujar las conexiones entre los landmarks de la cara
def dibujar_conexiones(frame, face_landmarks, mp_face_mesh):
    for connection in mp_face_mesh.FACEMESH_TESSELATION:
        start_idx, end_idx = connection
        start_point = face_landmarks.landmark[start_idx]
        end_point = face_landmarks.landmark[end_idx]
        start_pos = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
        end_pos = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
        cv2.line(frame, start_pos, end_pos, (0, 255, 0), 1)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Configuración de Mediapipe para la detección de la malla facial
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)  # Permitir hasta dos caras

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir frame.")
        break

    # Voltear horizontalmente la imagen
    frame = cv2.flip(frame, 1)

    # Convertir el frame a RGB (Mediapipe requiere RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar caras en el frame
    resultados = face_mesh.process(frame_rgb)

    # Extraer puntos clave de las caras detectadas
    if resultados.multi_face_landmarks:
        for face_landmarks in resultados.multi_face_landmarks:
            # Obtener los puntos clave de la cara con sus números
            puntos_cara = obtener_posicion_cara(face_landmarks, frame.shape)
            
            # Dibujar los puntos clave de la cara en el frame
            for (x, y, num) in puntos_cara:
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Dibujar un círculo en el punto

            # Mostrar el número de cada punto clave
            dibujar_numeros(frame, puntos_cara)

            # Dibujar conexiones entre los puntos de la cara
            dibujar_conexiones(frame, face_landmarks, mp_face_mesh)

    # Mostrar el frame con los puntos clave de la cara
    cv2.imshow('Detección de Cara', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()