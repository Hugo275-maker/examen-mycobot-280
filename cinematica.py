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

if __name__ == "__main__":
    fk = ForwardKinematics()
    fk.verify_home_pose()