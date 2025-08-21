
from matplotlib import pyplot as plt
import numpy as np
from skimage import draw


class CfdObstacle:
    def __init__(self):
        pass

    def get_mask(self, Xc, Yc, chi):
        pass

    def plot_obstacle(self, ax):
        pass


class CfdCircle(CfdObstacle):
    def __init__(self, radius=1, center=(0, 0)):
        super().__init__()
        self.radius = radius
        self.center = center

    def get_mask(self, Xc, Yc, chi):
        new_chi = chi.copy()
        cx, cy = self.center
        mask = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= self.radius ** 2
        new_chi[mask] = 0
        return new_chi
    
    def plot_obstacle(self, ax):
        circle = plt.Circle(self.center, self.radius, color='black', fill=True)
        ax.add_artist(circle)
        

class CfdTriangle(CfdObstacle):
    def __init__(self, vertices = np.array([[0, 0], [1, 0], [0, 1]])):
        super().__init__()
        self.vertices = vertices

    def get_mask(self, Xc, Yc, chi):
        new_chi = chi.copy()
        # Create a path for the triangle
        rr, cc = draw.polygon(self.vertices[:, 0], self.vertices[:, 1])
        new_chi[rr, cc] = 0
        return new_chi
    
    def plot_obstacle(self, ax):
        triangle = plt.Polygon(self.vertices, color='black', fill=True)
        ax.add_artist(triangle)


class CfdAirfoil(CfdObstacle):
    def __init__(self, NACA_code='0012'):
        super().__init__()
        self.NACA_code = NACA_code
        # Parse NACA 4-digit code
        self.m = int(self.NACA_code[0]) / 100.0  # maximum camber
        self.p = int(self.NACA_code[1]) / 10.0   # location of max camber
        self.t = int(self.NACA_code[2:]) / 100.0 # thickness
        x = np.linspace(0, 1, 200)
        self.x_upper, self.y_upper, self.x_lower, self.y_lower = self._upper_lower_surface(x)
        # Combine upper and lower surfaces
        self.x_coords = np.concatenate([self.x_upper[::-1], self.x_lower[1:]])
        self.y_coords = np.concatenate([self.y_upper[::-1], self.y_lower[1:]])


    def _airfoil_equation_thickness(self, x):
        yt = 5 * self.t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        return yt
    
    def _airfoil_equation_camber(self, x):
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        if self.m != 0 and self.p != 0:
            for i, xi in enumerate(x):
                if xi < self.p:
                    yc[i] = self.m / (self.p**2) * (2 * self.p * xi - xi**2)
                    dyc_dx[i] = 2 * self.m / (self.p**2) * (self.p - xi)
                else:
                    yc[i] = self.m / ((1 - self.p)**2) * ((1 - 2 * self.p) + 2 * self.p * xi - xi**2)
                    dyc_dx[i] = 2 * self.m / ((1 - self.p)**2) * (self.p - xi)
        return yc, dyc_dx
    
    def _upper_lower_surface(self, x):
        y_thickness = self._airfoil_equation_thickness(x)
        y_camber, dyc_dx = self._airfoil_equation_camber(x)
        theta = np.arctan(dyc_dx)
        x_upper = x - y_thickness * np.sin(theta)
        y_upper = y_camber + y_thickness * np.cos(theta)
        x_lower = x + y_thickness * np.sin(theta)
        y_lower = y_camber - y_thickness * np.cos(theta)
        return x_upper, y_upper, x_lower, y_lower

    def get_mask(self, Xc, Yc, chi):
        new_chi = chi.copy()     
        # Scale to grid
        x_idx = np.clip((self.x_coords * (Xc.shape[0] - 1)).astype(int), 0, Xc.shape[0] - 1)
        y_idx = np.clip((self.y_coords * (Yc.shape[1] - 1)).astype(int), 0, Yc.shape[1] - 1)
        rr, cc = draw.polygon(x_idx, y_idx)
        new_chi[rr, cc] = 0
        return new_chi
    
    def plot_obstacle(self, ax):
        ax.fill(self.x_coords, self.y_coords, color='black')
        ax.plot(self.x_upper, self.y_upper, color='black')
        ax.plot(self.x_lower, self.y_lower, color='black')