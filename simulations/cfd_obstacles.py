from matplotlib import pyplot as plt
import numpy as np
from matplotlib.path import Path


class CfdObstacle:
    """Base class for CFD obstacles."""
    def __init__(self):
        pass

    def get_mask(self, Xc: np.ndarray, Yc: np.ndarray, chi: np.ndarray) -> np.ndarray:
        """Get the obstacle mask.

        Args:
            Xc (np.ndarray): X-coordinates of the grid points.
            Yc (np.ndarray): Y-coordinates of the grid points.
            chi (np.ndarray): Original mask (1: fluid, 0: solid).

        Returns:
            np.ndarray: Updated mask with obstacle applied.
        """
        pass

    def plot_obstacle(self, ax: plt.Axes) -> None:
        """Plot the obstacle on the given axes.

        Args:
            ax (plt.Axes): Matplotlib axes to plot on.
        """
        pass

class EmptyObstacle(CfdObstacle):
    """No obstacle, entire domain is fluid."""
    def __init__(self):
        super().__init__()

    def get_mask(self, Xc: np.ndarray, Yc: np.ndarray, chi: np.ndarray) -> np.ndarray:
        return chi  # No obstacle, return original mask

    def plot_obstacle(self, ax: plt.Axes) -> None:
        pass  # Nothing to plot


class CfdCircle(CfdObstacle):
    """Circular obstacle."""
    def __init__(self, radius: float = 1.0, center: tuple = (0, 0)) -> None:
        super().__init__()
        self.radius = radius
        self.center = center

    def get_mask(self, Xc: np.ndarray, Yc: np.ndarray, chi: np.ndarray) -> np.ndarray:
        new_chi = chi.copy()
        cx, cy = self.center
        mask = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= self.radius ** 2
        new_chi[mask] = 0
        return new_chi

    def plot_obstacle(self, ax: plt.Axes) -> None:
        circle = plt.Circle(self.center, self.radius, color='black', fill=True)
        ax.add_artist(circle)
        
class CfdPolygon(CfdObstacle):
    """Polygonal obstacle defined by vertices."""
    def __init__(self, vertices: np.ndarray = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])) -> None:
        super().__init__()
        self.vertices = vertices

    def get_mask(self, Xc: np.ndarray, Yc: np.ndarray, chi: np.ndarray) -> np.ndarray:
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

    def plot_obstacle(self, ax: plt.Axes) -> None:
        """Plot polygon as a filled black shape"""
        polygon = plt.Polygon(self.vertices, color='black', fill=True)
        ax.add_patch(polygon)
        # Add outline for better visibility
        ax.plot(np.append(self.vertices[:, 0], self.vertices[0, 0]), 
                np.append(self.vertices[:, 1], self.vertices[0, 1]), 
                'k-', linewidth=1.5)
        
class CfdTriangle(CfdPolygon):
    """Triangular obstacle."""
    def __init__(self, vertices: np.ndarray = np.array([[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]])) -> None:
        super().__init__(vertices=vertices)


class CfdSquare(CfdPolygon):
    """Square obstacle."""
    def __init__(self, center: tuple = (0.5, 0.5), side_length: float = 0.2) -> None:
        half_side = side_length / 2
        vertices = np.array([
            [center[0] - half_side, center[1] - half_side],
            [center[0] + half_side, center[1] - half_side],
            [center[0] + half_side, center[1] + half_side],
            [center[0] - half_side, center[1] + half_side]
        ])
        super().__init__(vertices=vertices)


class CfdAirfoil(CfdObstacle):
    """ NACA 4-digit airfoil obstacle.
    Based on standard NACA equations for thickness and camber:
    self.NACA_code: The 4-digit string code (e.g., '0012') that defines the airfoil's shape according to NACA standards.
    self.center: Tuple (x, y) specifying the center position of the airfoil in the domain.
    self.chord_length: The length of the airfoil from leading to trailing edge (the “chord”).
    self.angle_of_attack: The orientation angle (in radians) of the airfoil relative to the horizontal axis.
    self.m: Maximum camber (curvature) as a fraction of the chord, parsed from the NACA code.
    self.p: Position of maximum camber along the chord, parsed from the NACA code.
    self.t: Maximum thickness as a fraction of the chord, parsed from the NACA code.
    self.x_coords, self.y_coords: Arrays of x and y coordinates outlining the airfoil's perimeter (used for masking and plotting).
    self.x_upper, self.y_upper, self.x_lower, self.y_lower: Arrays for the upper and lower surface coordinates, useful for detailed plotting.
    """
    def __init__(self, NACA_code: str = '0012', center: tuple = (0.5, 0.5), chord_length: float = 0.2, angle_of_attack: float = 0) -> None:
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

    def _airfoil_equation_thickness(self, x: np.ndarray) -> np.ndarray:
        """Calculate the airfoil thickness distribution.

        Args:
            x (np.ndarray): Array of x-coordinates (normalized).

        Returns:
            np.ndarray: Array of thickness values at each x-coordinate.
        """
        yt = 5 * self.t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        return yt

    def _airfoil_equation_camber(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the airfoil camber line and its slope.

        This function computes the camber line (mean line) and its derivative (slope) for a NACA 4-digit airfoil.
        The camber line describes the curvature of the airfoil, and its slope is used to determine the orientation
        of the upper and lower surfaces at each point along the chord.

        Args:
            x (np.ndarray): Array of x-coordinates (normalized, from 0 to 1 along the chord).

        Returns:
            tuple[np.ndarray, np.ndarray]: (y_camber, y_camber_slope) arrays, where y_camber is the camber line and y_camber_slope is its derivative with respect to x.
        """
        y_camber = np.zeros_like(x)
        y_camber_slope = np.zeros_like(x)
        if self.m != 0 and self.p != 0:
            for i, xi in enumerate(x):
                if xi < self.p:
                    y_camber[i] = self.m / (self.p**2) * (2 * self.p * xi - xi**2)
                    y_camber_slope[i] = 2 * self.m / (self.p**2) * (self.p - xi)
                else:
                    y_camber[i] = self.m / ((1 - self.p)**2) * ((1 - 2 * self.p) + 2 * self.p * xi - xi**2)
                    y_camber_slope[i] = 2 * self.m / ((1 - self.p)**2) * (self.p - xi)
        return y_camber, y_camber_slope

    def _upper_lower_surface(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the upper and lower surface coordinates of the airfoil.

        Args:
            x (np.ndarray): Array of x-coordinates (normalized).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (x_upper, y_upper, x_lower, y_lower) arrays for the positions of upper and lower surfaces.
        """
        y_thickness = self._airfoil_equation_thickness(x)
        y_camber, y_camber_slope = self._airfoil_equation_camber(x)
        theta = np.arctan(y_camber_slope)
        x_upper = x - y_thickness * np.sin(theta)
        y_upper = y_camber + y_thickness * np.cos(theta)
        x_lower = x + y_thickness * np.sin(theta)
        y_lower = y_camber - y_thickness * np.cos(theta)
        return x_upper, y_upper, x_lower, y_lower

    def get_mask(self, Xc: np.ndarray, Yc: np.ndarray, chi: np.ndarray) -> np.ndarray:
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
    
    def plot_obstacle(self, ax: plt.Axes) -> None:
        """Plot airfoil as a filled black shape"""
        ax.fill(self.x_coords, self.y_coords, color='black', zorder=10)
        ax.plot(self.x_coords, self.y_coords, color='black', linewidth=2, zorder=10)