import numpy as np
import math
import time
import matplotlib.pyplot as plt

try:
    from pymycobot.mycobot import MyCobot
    LIBRERIA_INSTALADA = True
except ModuleNotFoundError:
    LIBRERIA_INSTALADA = False
    print("Aviso: Librería 'pymycobot' no instalada. Ejecutando simulaciones locales...\n")

class ForwardKinematics:
    def dh_matrix(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])

    def compute_T0_6(self, thetas_deg):
        th = np.radians(thetas_deg)
        dh_params = [
            [th[0],             0.0,    0.0, np.radians(0)],
            [th[1],          134.65,    0.0, np.radians(90)],
            [th[2] - np.pi/2,   0.0, -110.0, np.radians(0)],
            [th[3],             0.0,  -96.0, np.radians(0)],
            [th[4] + np.pi/2,  63.4,    0.0, np.radians(-90)],
            [th[5] + np.pi/2, 75.05,   50.0, np.radians(90)]
        ]
        
        T_total = np.eye(4)
        for p in dh_params:
            T_total = np.dot(T_total, self.dh_matrix(p[0], p[1], p[2], p[3]))
        return T_total

    def verify_home_pose(self):
        T_home = self.compute_T0_6([0, 0, 0, 0, 0, 0])
        print("\n=== [P1] MATRIZ T0_6 EN POSE HOME ===")
        print(np.round(T_home, 2))

    def verify_fk_5_configs(self, mc=None):
        configuraciones = [
            [0, 0, 0, 0, 0, 0],
            [111.0, -24.69, -96.41, 38.14, 5.27, -8.34],
            [11.07, -15.64, -18.63, 39.9, 0.7, -32.51],
            [-73.91, -16.69, -102.74, 39.72, 0.17, -33.66],
            [111.0, -24.69, -96.41, 38.14, 5.27, -8.34]
        ]

        print("\n=== [P2] TABLA DE VALIDACIÓN FK (RUTINA REAL CAPTURADA) ===")
        print(f"{'Config (J1...J6)':<48} | {'Calc DH (X,Y,Z) mm':<22} | {'Real Robot':<18} | {'Error (mm)':<10}")
        print("-" * 105)

        for conf in configuraciones:
            T_calc = self.compute_T0_6(conf)
            x_c, y_c, z_c = T_calc[0,3], T_calc[1,3], T_calc[2,3]
            real_str, error_str = "N/A", "N/A"
            
            if mc:
                mc.send_angles(conf, 30)
                time.sleep(3.5) 
                pos_real = mc.get_coords()
                if pos_real:
                    x_r, y_r, z_r = pos_real[0], pos_real[1], pos_real[2]
                    real_str = f"({x_r:.1f}, {y_r:.1f}, {z_r:.1f})"
                    error = np.sqrt((x_c - x_r)**2 + (y_c - y_r)**2 + (z_c - z_r)**2)
                    error_str = f"{error:.2f}"

            conf_str = f"[{conf[0]:.1f}, {conf[1]:.1f}, {conf[2]:.1f}, {conf[3]:.1f}, {conf[4]:.1f}, {conf[5]:.1f}]"
            print(f"{conf_str:<48} | ({x_c:.1f}, {y_c:.1f}, {z_c:.1f}) | {real_str:<18} | {error_str:<10}")

    def plot_workspace_xz(self, num_points=5000):
        print("\n[P2] Generando gráfica del Espacio de Trabajo (Plano XZ)...")
        X_vals, Z_vals = [], []
        for _ in range(num_points):
            t1 = 0 
            t2 = np.random.uniform(-135, 90)
            t3 = np.random.uniform(-150, 150)
            t4 = np.random.uniform(-145, 145)
            t5 = np.random.uniform(-165, 165)
            t6 = 0  
            T = self.compute_T0_6([t1, t2, t3, t4, t5, t6])
            X_vals.append(T[0,3])
            Z_vals.append(T[2,3])
            
        plt.figure(figsize=(8, 8))
        plt.scatter(X_vals, Z_vals, s=2, alpha=0.3, color='royalblue')
        plt.title('Espacio de Trabajo Alcanzable - MyCobot 280 (Plano XZ)')
        plt.xlabel('Eje X (mm) - Alcance Radial')
        plt.ylabel('Eje Z (mm) - Altura')
        plt.grid(True, linestyle='--')
        plt.axis('equal') 
        plt.show()

class InverseKinematics:
    def __init__(self):
        self.A2 = 110.0
        self.A3 = 96.0
        self.D1 = 134.65

    def ik_solve(self, x, y, z):
        j1_rad = math.atan2(y, x)
        r = math.sqrt(x**2 + y**2)
        s = z - self.D1
        
        num = (r**2 + s**2 - self.A2**2 - self.A3**2)
        den = (2 * self.A2 * self.A3)
        cos_j3 = num / den
        
        if cos_j3 > 1.0 or cos_j3 < -1.0:
            return None 
            
        sin_j3 = -math.sqrt(1 - cos_j3**2)
        j3_rad = math.atan2(sin_j3, cos_j3)
        
        alpha = math.atan2(s, r)
        beta = math.atan2(self.A3 * sin_j3, self.A2 + self.A3 * cos_j3)
        j2_rad = alpha - beta
        
        return [math.degrees(j1_rad), math.degrees(j2_rad), math.degrees(j3_rad)]

    def verify_ik_3_positions(self, mc=None):
        posiciones_target = [
            [-18.0, 214.6, 142.8],
            [134.9, -38.8, 394.4],
            [-7.6, -206.1, 155.0]
        ]
        
        print("\n=== [P3] TABLA DE VALIDACIÓN IK (COORDENADAS REALES) ===")
        print(f"{'Target (X,Y,Z) mm':<22} | {'IK Analítica (J1,J2,J3)':<30} | {'IK Real Robot':<18} | {'Error Prom (°)':<10}")
        print("-" * 90)
        
        for pos in posiciones_target:
            x, y, z = pos[0], pos[1], pos[2]
            sol_analitica = self.ik_solve(x, y, z)
            
            if sol_analitica is None:
                print(f"{str(pos):<22} | Fuera de rango matemático")
                continue
                
            analitica_str = f"({sol_analitica[0]:.1f}°, {sol_analitica[1]:.1f}°, {sol_analitica[2]:.1f}°)"
            real_str, error_str = "N/A", "N/A"
            
            if mc:
                mc.send_coords([x, y, z, 0, 0, 0], 30, 1)
                time.sleep(4.0) 
                angulos_real = mc.get_angles()
                if angulos_real and len(angulos_real) >= 3:
                    real_str = f"({angulos_real[0]:.1f}°, {angulos_real[1]:.1f}°, {angulos_real[2]:.1f}°)"
                    err = (abs(sol_analitica[0] - angulos_real[0]) + 
                           abs(sol_analitica[1] - angulos_real[1]) + 
                           abs(sol_analitica[2] - angulos_real[2])) / 3.0
                    error_str = f"{err:.2f}°"
                
                mc.send_angles([11.07, -15.64, -18.63, 39.9, 0.7, -32.51], 30)
                time.sleep(3.0)
                    
            print(f"{str(pos):<22} | {analitica_str:<30} | {real_str:<18} | {error_str:<10}")

class CollisionChecker:
    def __init__(self, fk_solver):
        self.fk = fk_solver
        self.Z_SEGURIDAD_MM = 50.0 
        self.X_SEGURIDAD_MM = -150.0
        self.POSE_TRANSITO = [11.07, -15.64, -18.63, 39.9, 0.7, -32.51] 

    def safe_move(self, mc, target_angles, speed=20):
        print(f"\n[Seguridad] Evaluando movimiento hacia: {[round(a,1) for a in target_angles]}")
        T_futura = self.fk.compute_T0_6(target_angles)
        x_esperada = T_futura[0, 3]
        z_esperada = T_futura[2, 3]
        
        if z_esperada < self.Z_SEGURIDAD_MM:
            print(f" 🛑 BLOQUEO INMEDIATO: Peligro de impacto con la mesa (Z={z_esperada:.1f} mm).")
            return False
            
        if x_esperada < self.X_SEGURIDAD_MM:
            print(f" 🛑 BLOQUEO INMEDIATO: Peligro de impacto con pedestal/cables (X={x_esperada:.1f} mm).")
            return False

        print(f" ✅ GEOFENCING OK: Coordenadas proyectadas dentro de la caja de seguridad.")

        if not LIBRERIA_INSTALADA or mc is None:
            return True

        pos_actual = mc.get_angles()
        if pos_actual and len(pos_actual) > 0:
            dif_j1 = abs(pos_actual[0] - target_angles[0])
            if dif_j1 > 60:
                print(" 🌉 ALERTA BARRIDO: Trayectoria larga detectada. Inyectando Waypoint MEDIO...")
                mc.send_angles(self.POSE_TRANSITO, speed)
                time.sleep(3.5)
        
        print(" 🚀 Ejecutando movimiento al objetivo...")
        mc.send_angles(target_angles, speed)
        time.sleep(3)
        return True

    def run_collision_tests(self, mc=None):
        print("\n=== [P4] RUTINA DE VALIDACIÓN ANTI-COLISIONES ===")
        POSE_INICIO = [111.0, -24.69, -96.41, 38.14, 5.27, -8.34]
        POSE_FINAL  = [-73.91, -16.69, -102.74, 39.72, 0.17, -33.66]
        POSE_CHOQUE = [0, 90, 45, 0, 0, 0] 
        POSE_CHOQUE_TRASERO = [180, -45, 0, 0, 0, 0]
        
        if mc:
            mc.send_angles(self.POSE_TRANSITO, 30)
            time.sleep(3)

        print("\nPrueba 1: Movimiento Seguro a Inicio")
        self.safe_move(mc, POSE_INICIO)
        
        print("\nPrueba 2: Movimiento Cruzado a Final (Activará Waypoint)")
        self.safe_move(mc, POSE_FINAL)
        
        print("\nPrueba 3: Intento de Penetración de Mesa (Límite Z)")
        self.safe_move(mc, POSE_CHOQUE)
        
        print("\nPrueba 4: Intento de Colisión Trasera (Límite X)")
        self.safe_move(mc, POSE_CHOQUE_TRASERO)

if __name__ == "__main__":
    fk = ForwardKinematics()
    ik = InverseKinematics()
    seguridad = CollisionChecker(fk)
    
    robot = None
    if LIBRERIA_INSTALADA:
        try:
            robot = MyCobot('/dev/ttyUSB0', 1000000)
            robot.power_on()
            time.sleep(1)
            if not robot.is_controller_connected():
                robot = None
            else:
                print("\n[Sistema] Hardware MyCobot conectado y verificado.")
        except Exception:
            robot = None

    print("\n" + "="*50)
    print(" INICIANDO BATERÍA DE PRUEBAS DEL EXAMEN GRUPAL")
    print("="*50)
            
    fk.verify_home_pose()                   
    fk.verify_fk_5_configs(mc=robot)        
    ik.verify_ik_3_positions(mc=robot)      
    seguridad.run_collision_tests(mc=robot) 
    
    if robot:
        print("\n[Sistema] Pruebas finalizadas. Evaluando Gripper...")
        robot.set_gripper_state(0, 50) 
        time.sleep(2)
        robot.set_gripper_state(1, 50) 
        time.sleep(2)
        
        print("[Sistema] Regresando a posición INICIO de descanso...")
        robot.send_angles([111.0, -24.69, -96.41, 38.14, 5.27, -8.34], 20)
        time.sleep(3.5)
        print("[Sistema] Robot aparcado de forma segura y apagado de motores.")
        robot.release_all_servos() 
    
    fk.plot_workspace_xz()