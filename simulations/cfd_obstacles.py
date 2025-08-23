from matplotlib import pyplot as plt
import numpy as np
from matplotlib.path import Path


class CfdObstacle:
    def __init__(self):
        pass

    def get_mask(self, Xc, Yc, chi):
        pass

    def plot_obstacle(self, ax):
        pass

class EmptyObstacle(CfdObstacle):
    def __init__(self):
        super().__init__()

    def get_mask(self, Xc, Yc, chi):
        return chi  # No obstacle, return original mask

    def plot_obstacle(self, ax):
        pass  # Nothing to plot


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
        
class CfdPolygon(CfdObstacle):
    def __init__(self, vertices=np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])):
        super().__init__()
        self.vertices = vertices

    def get_mask(self, Xc, Yc, chi):
        """Create mask for polygon obstacle using matplotlib.path"""
        new_chi = chi.copy()
        
        polygon_path = Path(self.vertices)
        
        # Check which grid points are inside the polygon
        points = np.column_stack((Xc.ravel(), Yc.ravel()))
        inside = polygon_path.contains_points(points)
        inside_mask = inside.reshape(Xc.shape)
        
        # Set obstacle cells to 0 (solid)
        new_chi[inside_mask] = 0
        return new_chi
    
    def plot_obstacle(self, ax: plt.Axes):
        """Plot polygon as a filled black shape"""
        polygon = plt.Polygon(self.vertices, color='black', fill=True)
        ax.add_patch(polygon)
        # Add outline for better visibility
        ax.plot(np.append(self.vertices[:, 0], self.vertices[0, 0]), 
                np.append(self.vertices[:, 1], self.vertices[0, 1]), 
                'k-', linewidth=1.5)
        
class CfdTriangle(CfdPolygon):
    def __init__(self, vertices=np.array([[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]])):
        super().__init__(vertices=vertices)


class CfdSquare(CfdPolygon):
    def __init__(self, center=(0.5, 0.5), side_length=0.2):
        half_side = side_length / 2
        vertices = np.array([
            [center[0] - half_side, center[1] - half_side],
            [center[0] + half_side, center[1] - half_side],
            [center[0] + half_side, center[1] + half_side],
            [center[0] - half_side, center[1] + half_side]
        ])
        super().__init__(vertices=vertices)


class CfdAirfoil(CfdObstacle):
    def __init__(self, NACA_code='0012', center=(0.5, 0.5), chord_length=0.2, angle_of_attack=0):
        super().__init__()
        self.NACA_code = NACA_code
        self.center = center
        self.chord_length = chord_length
        self.angle_of_attack = np.radians(angle_of_attack)  # Convert to radians
        
        # Parse NACA 4-digit code
        self.m = int(self.NACA_code[0]) / 100.0  # maximum camber
        self.p = int(self.NACA_code[1]) / 10.0   # location of max camber
        self.t = int(self.NACA_code[2:]) / 100.0 # thickness
        
        # Generate airfoil coordinates
        x = np.linspace(0, 1, 200)
        x_upper, y_upper, x_lower, y_lower = self._upper_lower_surface(x)
        
        # Combine upper and lower surfaces (close the airfoil)
        x_coords = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        # Scale by chord length
        x_coords *= self.chord_length
        y_coords *= self.chord_length
        
        # Rotate by angle of attack
        if self.angle_of_attack != 0:
            cos_alpha = np.cos(self.angle_of_attack)
            sin_alpha = np.sin(self.angle_of_attack)
            x_rot = x_coords * cos_alpha - y_coords * sin_alpha
            y_rot = x_coords * sin_alpha + y_coords * cos_alpha
            x_coords, y_coords = x_rot, y_rot
        
        # Translate to center position
        self.x_coords = x_coords + self.center[0]
        self.y_coords = y_coords + self.center[1]
        
        # Store upper and lower surfaces for plotting
        self.x_upper = x_upper * self.chord_length + self.center[0]
        self.y_upper = y_upper * self.chord_length + self.center[1]
        self.x_lower = x_lower * self.chord_length + self.center[0]
        self.y_lower = y_lower * self.chord_length + self.center[1]

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
        """Create mask for airfoil obstacle in the CFD grid"""
        new_chi = chi.copy()
        
        # Create polygon mask using the airfoil coordinates
        airfoil_path = Path(np.column_stack((self.x_coords, self.y_coords)))
        
        # Check which grid points are inside the airfoil
        points = np.column_stack((Xc.ravel(), Yc.ravel()))
        inside = airfoil_path.contains_points(points)
        inside_mask = inside.reshape(Xc.shape)
        
        # Set obstacle cells to 0
        new_chi[inside_mask] = 0
        return new_chi
    
    def plot_obstacle(self, ax: plt.Axes):
        """Plot airfoil as a filled black shape"""
        ax.fill(self.x_coords, self.y_coords, color='black', zorder=10)
        ax.plot(self.x_coords, self.y_coords, color='black', linewidth=2, zorder=10)