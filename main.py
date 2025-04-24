#Importar librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import math

# Cargar el video
video_path = 'short.mp4'
#video_path = 'long.mp4'
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames por segundo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS: {fps}, Total Frames: {total_frames}, Resolution: {width}x{height}")

# Verificar que el video se abrió correctamente
if not cap.isOpened():
    print('Error opening video file')
    exit()
else:
    print('Video file opened')

# Variables para redimensionar los frames
new_height = 480
new_width = 680

# Establecer punto de sujeción fijo en (344, 0)
anchor_x = 344
anchor_y = 0
print(f"Punto de sujeción fijo: ({anchor_x}, {anchor_y})")

# Variable para almacenar la historia del centroide
centroid_history = []  # Lista para almacenar (tiempo, x, y, ángulo)
pixel_diameter = None  # Para calibración
real_diameter_cm = 2.5  # Diámetro real de la masa en cm
cord_length_cm = 10.3  # Longitud de la cuerda en cm
pixels_per_cm = None  # Factor de conversión píxeles a cm

# Para visualización de la trayectoria
trajectory_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

# Contador de frames y tiempo
frame_count = 0
time_count = 0.0

# Variables para cálculo de velocidad y aceleración en tiempo real
prev_x_cm, prev_y_cm = None, None
prev_vx, prev_vy = None, None
current_vx, current_vy = 0, 0
current_ax, current_ay = 0, 0
prev_time = 0

angle_history = []  # Lista para almacenar el historial de ángulos

# Variable para guardar los datos para análisis posterior
all_data = {
    'time': [],
    'x_px': [], 'y_px': [],
    'x_cm': [], 'y_cm': [],
    'vx': [], 'vy': [], 'v_total': [],
    'ax': [], 'ay': [], 'a_total': [],
    'angle': []
}

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Salir del bucle si no hay más frames

    # Incrementar contador de frames y tiempo
    frame_count += 1
    time_count = frame_count / fps

    # Redimensionar el frame
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Convertir a HSV para mejor segmentación por color
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Definir los límites para la segmentación
    lower_limit = np.array([0, 0, 0])
    upper_limit = np.array([35, 255, 255])

    # Crear la máscara binaria
    mask = cv2.inRange(hsv, lower_limit, upper_limit)

    # Limpiar la máscara usando operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=2)

    # Crear una copia del frame para dibujar
    display_frame = frame_resized.copy()

    # Encontrar contornos en la máscara limpia
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar el centroide si se encuentra un contorno
    if contours:
        # Encontrar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        # Calcular diámetro aproximado para calibración
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        current_diameter = 2 * radius

        # Si es la primera detección, usarla para calibración
        if pixel_diameter is None:
            pixel_diameter = current_diameter
            pixels_per_cm = pixel_diameter / real_diameter_cm
            print(f"Calibración: {pixels_per_cm:.2f} píxeles por cm")

        # Calcular los momentos del contorno
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:  # Evitar división por cero
            # Calcular las coordenadas del centroide
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calcular el ángulo del péndulo (en radianes, luego convertido a grados)
            # El ángulo 0 corresponde a la posición vertical hacia abajo
            dx = cx - anchor_x
            dy = cy - anchor_y
            angle_rad = math.atan2(dx, dy)
            angle_deg = math.degrees(angle_rad)

            # Almacenar ángulo en el historial
            angle_history.append(angle_deg)

            # Almacenar datos del centroide (tiempo, x, y)
            centroid_history.append((time_count, cx, cy, angle_deg))

            # Dibujar el contorno del objeto
            cv2.drawContours(display_frame, [largest_contour], -1, (255, 0, 0), 2)  # Contorno azul

            # Dibujar el centroide en el frame
            cv2.circle(display_frame, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

            # Dibujar la línea del péndulo
            cv2.line(display_frame, (anchor_x, anchor_y), (cx, cy), (0, 0, 255), 2)
            cv2.circle(display_frame, (anchor_x, anchor_y), radius=3, color=(255, 0, 0), thickness=-1)

            # Mostrar información en tiempo real
            if pixels_per_cm is not None:
                # Convertir posición a cm
                x_cm = cx / pixels_per_cm
                y_cm = cy / pixels_per_cm

                # Guardar datos para análisis
                all_data['time'].append(time_count)
                all_data['x_px'].append(cx)
                all_data['y_px'].append(cy)
                all_data['x_cm'].append(x_cm)
                all_data['y_cm'].append(y_cm)
                all_data['angle'].append(angle_deg)

                # Calcular velocidad en tiempo real si tenemos posición previa
                if prev_x_cm is not None and (time_count - prev_time) > 0:
                    dt = time_count - prev_time
                    current_vx = (x_cm - prev_x_cm) / dt
                    current_vy = (y_cm - prev_y_cm) / dt

                    # Guardar datos de velocidad
                    all_data['vx'].append(current_vx)
                    all_data['vy'].append(current_vy)
                    all_data['v_total'].append(math.sqrt(current_vx ** 2 + current_vy ** 2))

                    # Calcular aceleración en tiempo real si tenemos velocidad previa
                    if prev_vx is not None:
                        current_ax = (current_vx - prev_vx) / dt
                        current_ay = (current_vy - prev_vy) / dt

                        # Guardar datos de aceleración
                        all_data['ax'].append(current_ax)
                        all_data['ay'].append(current_ay)
                        all_data['a_total'].append(math.sqrt(current_ax ** 2 + current_ay ** 2))
                    else:
                        all_data['ax'].append(0)
                        all_data['ay'].append(0)
                        all_data['a_total'].append(0)
                else:
                    all_data['vx'].append(0)
                    all_data['vy'].append(0)
                    all_data['v_total'].append(0)
                    all_data['ax'].append(0)
                    all_data['ay'].append(0)
                    all_data['a_total'].append(0)

                # Guardar valores actuales para el siguiente frame
                prev_x_cm, prev_y_cm = x_cm, y_cm
                prev_vx, prev_vy = current_vx, current_vy
                prev_time = time_count

                # Calcular magnitudes totales
                current_v_total = math.sqrt(current_vx ** 2 + current_vy ** 2)
                current_a_total = math.sqrt(current_ax ** 2 + current_ay ** 2)

                # Mostrar información en el frame
                # Tiempo
                cv2.putText(display_frame, f"T: {time_count:.2f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Posición
                cv2.putText(display_frame, f"X: {cx} px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Y: {cy} px", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Velocidad
                cv2.putText(display_frame, f"V: {current_v_total:.2f} cm/s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                # Aceleración
                cv2.putText(display_frame, f"A: {current_a_total:.2f} cm/s²", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # Ángulo
                cv2.putText(display_frame, f"Angulo: {angle_deg:.2f}°", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Imprimir datos en consola (cada 5 frames para no saturar)
                if frame_count % 5 == 0:
                    print(f"\n--- Frame {frame_count} (T={time_count:.3f}s) ---")
                    print(f"Posición: X={cx} px, Y={cy} px")
                    print(
                        f"Velocidad: Vx={current_vx:.2f} cm/s, Vy={current_vy:.2f} cm/s, |V|={current_v_total:.2f} cm/s")
                    print(
                        f"Aceleración: Ax={current_ax:.2f} cm/s², Ay={current_ay:.2f} cm/s², |A|={current_a_total:.2f} cm/s²")
                    print(f"Ángulo: {angle_deg:.2f}°")

            # Dibujar en la imagen de trayectoria
            if len(centroid_history) >= 2:
                # Obtener punto previo
                prev_t, prev_x, prev_y, _ = centroid_history[-2]
                # Dibujar línea entre puntos consecutivos
                cv2.line(trajectory_image, (prev_x, prev_y), (cx, cy), (0, 255, 255), 2)

    # Superponer trayectoria en el frame con cierta transparencia
    display_with_trajectory = cv2.addWeighted(display_frame, 0.7, trajectory_image, 0.3, 0)

    # Mostrar frames
    cv2.imshow('Tracking', display_with_trajectory)
    # cv2.imshow('Mask', mask_cleaned)

    # Salir del bucle si se presiona la tecla ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Liberar recursos de video
cap.release()
cv2.destroyAllWindows()

# --- Análisis posterior de los datos ---

if len(centroid_history) > 10:  # Si tenemos suficientes datos
    # Extraer datos para análisis
    times = all_data['time']
    x_positions = all_data['x_px']
    y_positions = all_data['y_px']
    angles = all_data['angle']

    # Convertir a arrays de numpy
    times = np.array(times)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    angles = np.array(angles)

    # Datos de velocidad y aceleración
    v_total = np.array(all_data['v_total'][1:])  # Excluimos el primer valor que es 0
    a_total = np.array(all_data['a_total'][2:])  # Excluimos los dos primeros valores

    # Tiempos para velocidad y aceleración (ajustar para que coincidan con los datos)
    times_v = times[1:]
    times_a = times[2:]

    # Suavizar datos con filtro Savitzky-Golay para reducir ruido
    window_length = min(51, len(times) - (len(times) % 2 - 1))
    if window_length > 3:
        x_smooth = savgol_filter(x_positions, window_length, 3)
        y_smooth = savgol_filter(y_positions, window_length, 3)
        angles_smooth = savgol_filter(angles, window_length, 3)
    else:
        x_smooth = x_positions
        y_smooth = y_positions
        angles_smooth = angles

    # Suavizar velocidad y aceleración
    if len(v_total) > 5:
        window_length_v = min(21, len(v_total) - (len(v_total) % 2 - 1))
        if window_length_v > 3:
            v_total_smooth = savgol_filter(v_total, window_length_v, 3)
        else:
            v_total_smooth = v_total
    else:
        v_total_smooth = v_total

    if len(a_total) > 5:
        window_length_a = min(21, len(a_total) - (len(a_total) % 2 - 1))
        if window_length_a > 3:
            a_total_smooth = savgol_filter(a_total, window_length_a, 3)
        else:
            a_total_smooth = a_total
    else:
        a_total_smooth = a_total

    # Buscar picos en la posición X o ángulo para estimar el periodo
    peaks, _ = find_peaks(angles_smooth, height=0)

    if len(peaks) >= 2:
        # Calcular periodos entre picos consecutivos
        peak_times = times[peaks]
        periods = np.diff(peak_times)
        avg_period = np.mean(periods)

        print(f"\n--- RESULTADOS FINALES ---")
        print(f"Periodo estimado: {avg_period:.4f} segundos")

        # Calcular la longitud efectiva del péndulo usando T = 2π√(L/g)
        # T = 2π√(L/g) → L = g*(T/(2π))²
        g = 9.81  # m/s²
        L_estimated = g * (avg_period / (2 * np.pi)) ** 2 * 100  # en cm

        print(f"Longitud estimada del péndulo: {L_estimated:.2f} cm")
        print(f"Longitud real del péndulo (cuerda + radio): {cord_length_cm + real_diameter_cm / 2:.2f} cm")

        # Calcular frecuencia y frecuencia angular
        freq = 1 / avg_period
        omega = 2 * np.pi * freq

        print(f"Frecuencia: {freq:.4f} Hz")
        print(f"Frecuencia angular: {omega:.4f} rad/s")

        # Calcular amplitud máxima en grados
        amplitude_deg = np.max(np.abs(angles_smooth))
        print(f"Amplitud máxima: {amplitude_deg:.2f}°")

    # Crear gráficas combinadas
    plt.figure(figsize=(15, 16))

    # 1. Posición vs Tiempo
    plt.subplot(4, 1, 1)
    plt.plot(times, x_smooth, 'b-', label='X (px)')
    plt.plot(times, y_smooth, 'g-', label='Y (px)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (píxeles)')
    plt.title('Posición vs Tiempo')
    plt.grid(True)
    plt.legend()

    # 2. Ángulo vs Tiempo
    plt.subplot(4, 1, 2)
    plt.plot(times, angles_smooth, 'm-', label='Ángulo (grados)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (°)')
    plt.title('Ángulo vs Tiempo')
    plt.grid(True)
    plt.legend()

    # 3. Velocidad vs Tiempo
    plt.subplot(4, 1, 3)
    plt.plot(times_v, v_total_smooth, 'r-', label='V total (cm/s)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (cm/s)')
    plt.title('Velocidad vs Tiempo')
    plt.grid(True)
    plt.legend()

    # 4. Aceleración vs Tiempo
    plt.subplot(4, 1, 4)
    plt.plot(times_a, a_total_smooth, 'k-', label='A total (cm/s²)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración (cm/s²)')
    plt.title('Aceleración vs Tiempo')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No se recolectaron suficientes datos para el análisis.")