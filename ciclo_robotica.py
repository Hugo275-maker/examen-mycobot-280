import cv2
import numpy as np
import time

# --- MODULO DE SIMULACION (P5) ---
class SimulatedMyCobot:
    def __init__(self, port, baud):
        print(f"[SIMULADOR] Robot inicializado en {port}")
    def power_on(self):
        print("[SIMULADOR] Motores encendidos")
    def send_angles(self, angles, speed):
        print(f"[SIMULADOR] Moviendo a ángulos: {angles} (Vel: {speed})")
    def send_coords(self, coords, speed, mode):
        print(f"[SIMULADOR] Moviendo a coord: {coords} (Modo: {mode})")
    def set_gripper_state(self, state, speed):
        estado = "CERRADO" if state == 1 else "ABIERTO"
        print(f"[SIMULADOR] Gripper: {estado}")
    def is_controller_connected(self):
        return True

# Intentar cargar librería real, si falla, usamos el simulador
try:
    from pymycobot.mycobot import MyCobot
    mc = MyCobot('/dev/ttyUSB0', 1000000)
    if not mc.is_controller_connected():
        raise Exception("No detectado")
    mc.power_on()
except:
    print("!!! Hardware no detectado: Activando MODO SIMULACIÓN !!!")
    mc = SimulatedMyCobot('SIMULADO', 1000000)

# ==========================================
# [P5 & P6] INTEGRACIÓN: CONTROL Y VISIÓN
# ==========================================

INIT_POSE = [0, 0, 0, 0, 0, -45]
WATCH_POSE = [11.07, -20, -18.63, 39.9, 0.7, -32.51]

def capturar_centroide():
    """Lógica de Visión (P6.2): Detección por color HSV"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    # Filtro HSV (P6.1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 50, 50])
    hsv_max = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # Limpieza (P6.5)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(c) > 500: # Filtro de ruido
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cap.release()
            return (cx, cy)
    
    cap.release()
    return None

def ejecutar_ciclo_agarre():
    """Control de trayectoria (P5.2)"""
    print("--- Ciclo de Agarre Iniciado ---")
    mc.send_angles(WATCH_POSE, 30)
    time.sleep(2)
    
    coord = capturar_centroide()
    if coord:
        # Transformación (P6.3) - Ajustar estos factores en Lab
        x_real = (coord[0] - 320) * 0.4
        y_real = (coord[1] - 240) * 0.4
        
        print(f"Objeto localizado. Moviendo a: {x_real}, {y_real}")
        mc.send_coords([x_real, y_real, 100, 0, 0, 0], 30, 1)
        time.sleep(2)
        mc.set_gripper_state(1, 50) # Cerrar
        time.sleep(2)
        mc.send_coords([100, 100, 100, 0, 0, 0], 30, 1)
        mc.set_gripper_state(0, 50) # Abrir
        return True
    return False

if __name__ == "__main__":
    mc.send_angles(INIT_POSE, 30)
    time.sleep(2)
    
    # Ejecución de 5 ciclos (P5.4)
    for i in range(5):
        print(f"Ciclo {i+1} de 5")
        ejecutar_ciclo_agarre()
    
    mc.send_angles(INIT_POSE, 30)
    print("Proceso finalizado.")