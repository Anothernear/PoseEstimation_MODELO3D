import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --- CONFIGURACIÓN DE MEDIAPIPE Y OPENCV ---
map_hands = mp.solutions.hands
hand_detector = map_hands.Hands()
map_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
frame_width, frame_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# --- VARIABLES DE ESTADO Y UMBRALES ---
is_mouse_down = False
CLOSED_HAND_THRESHOLD = 0.10
OPEN_HAND_THRESHOLD = 0.10
CLICK_FRAMES_THRESHOLD = 5
RELEASE_FRAMES_THRESHOLD = 5
SCROLL_SENSITIVITY = 2  # Cantidad de scroll a realizar (ajustable)
MIN_DISTANCE_FOR_SCROLL = 0.05  # Distancia mínima para empezar a considerar el scroll

initial_distance_threshold = None 
click_frames_count = 0
release_frames_count = 0

# Variables para el movimiento relativo y el filtro.
prev_finger_x, prev_finger_y = None, None 
SENSITIVITY = 2.0 
MAX_MOVEMENT_THRESHOLD = 50 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(frame_rgb)
    # --- Lógica de detección y cálculo ---
    left_wrist_pos = None
    right_wrist_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 2. Obtener la etiqueta de la mano ("Left" o "Right")
            hand_label = handedness.classification[0].label
        
            map_drawing.draw_landmarks(
                frame, hand_landmarks, map_hands.HAND_CONNECTIONS,
                map_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                map_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            index_finger_tip = hand_landmarks.landmark[map_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # --- LÓGICA DE MOVIMIENTO RELATIVO CON FILTRO ---
            current_x_norm = index_finger_tip.x
            current_y_norm = index_finger_tip.y
            
            if prev_finger_x is not None:
                delta_x = (current_x_norm - prev_finger_x) * screen_width * SENSITIVITY
                delta_y = (current_y_norm - prev_finger_y) * screen_height * SENSITIVITY
                
                if abs(delta_x) < MAX_MOVEMENT_THRESHOLD and abs(delta_y) < MAX_MOVEMENT_THRESHOLD:
                    # Mueve el cursor usando pyautogui (movimiento relativo)
                    pyautogui.moveRel(delta_x, delta_y)
                else:
                    print(f"Movimiento filtrado: Salto detectado ({delta_x:.2f}, {delta_y:.2f})")
            
            prev_finger_x, prev_finger_y = current_x_norm, current_y_norm

            # --- LÓGICA DE CLIC POR CIERRE DE MANO ---
            wrist = hand_landmarks.landmark[map_hands.HandLandmark.WRIST]
            fingertips = [
                hand_landmarks.landmark[map_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[map_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[map_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[map_hands.HandLandmark.PINKY_TIP],
            ]
            
            total_distance = 0
            for tip in fingertips:
                dist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                total_distance += dist
            
            average_distance = total_distance / len(fingertips)
            cv2.putText(frame, f"Distancia: {average_distance:.2f}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 40), 2, cv2.LINE_AA)

            if average_distance < CLOSED_HAND_THRESHOLD:
                click_frames_count += 1
                release_frames_count = 0
            elif average_distance > OPEN_HAND_THRESHOLD:
                release_frames_count += 1
                click_frames_count = 0
            else:
                pass
            
            if not is_mouse_down and click_frames_count >= CLICK_FRAMES_THRESHOLD:
                # Presiona el botón del mouse usando pyautogui
                pyautogui.mouseDown()
                is_mouse_down = True
                print("Mouse presionado (arrastre activado)")
            elif is_mouse_down and release_frames_count >= RELEASE_FRAMES_THRESHOLD:
                # Suelta el botón del mouse usando pyautogui
                pyautogui.mouseUp()
                is_mouse_down = False
                print("Mouse soltado")

            x_finger_pixel = int(index_finger_tip.x * frame_width)
            y_finger_pixel = int(index_finger_tip.y * frame_height)
            if is_mouse_down:
                cv2.circle(frame, (x_finger_pixel, y_finger_pixel), 25, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x_finger_pixel, y_finger_pixel), 15, (0, 255, 255), -1)
            

            #--LOGICA DE ZOOM_IN & ZOOM_OUT--#

            # Almacenar la posición de cada muñeca
            if hand_label == 'Right':
                right_wrist_pos = wrist
            elif hand_label == 'Left':
                left_wrist_pos = wrist
            
            if left_wrist_pos and right_wrist_pos:
                axis_x = right_wrist_pos.x - left_wrist_pos.x
                axis_y = right_wrist_pos.y - left_wrist_pos.y
                distance = math.hypot(axis_x, axis_y)

                # Para la calibración inicial, si es la primera vez que detectamos ambas manos
                if initial_distance_threshold is None and distance > MIN_DISTANCE_FOR_SCROLL:
                    initial_distance_threshold = distance
                    print(f"Distancia inicial de calibración establecida: {initial_distance_threshold:.2f}")

                # Solo si la distancia inicial ya está calibrada
                if initial_distance_threshold is not None:
                    # Comprobar si la distancia ha cambiado significativamente
                    # Se usa una tolerancia para evitar scrolls accidentales por pequeñas variaciones
                    if distance > initial_distance_threshold * 1.15: # Aumento del 15%
                        print(f"Distancia aumentada: {distance:.2f}. Simulando Zoom IN (scroll up).")
                        pyautogui.scroll(SCROLL_SENSITIVITY) # Scroll hacia arriba
                        # Actualizar el umbral para evitar scroll continuo
                        initial_distance_threshold = distance 
                    elif distance < initial_distance_threshold * 0.85 and distance > MIN_DISTANCE_FOR_SCROLL: # Disminución del 15%
                        print(f"Distancia disminuida: {distance:.2f}. Simulando Zoom OUT (scroll down).")
                        pyautogui.scroll(-SCROLL_SENSITIVITY) # Scroll hacia abajo
                        # Actualizar el umbral para evitar scroll continuo
                        initial_distance_threshold = distance 
                cv2.putText(frame, f"Distancia: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        prev_finger_x, prev_finger_y = None, None

    cv2.imshow('Hand Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()