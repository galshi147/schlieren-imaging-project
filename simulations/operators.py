import numpy as np

class FiniteDifferenceOperators:
    """
    A utility class for finite difference operators on staggered grids.
    The class provides implementations on a 2D MAC (Marker-And-Cell) staggered grid
    of vector calculus operators in cartesian coordinates:
    - gradient (partial derivatives)
    - divergence
    - laplacian
    - interpolation between staggered grid components
    """
    def __init__(self):
        pass
    
    @staticmethod
    def gradient_x(field: np.ndarray, dx: float) -> np.ndarray:
        """Compute gradient in x-direction using central differences

        Args:
            field (np.ndarray): field to differentiate
            dx (float): grid spacing in x-direction

        Returns:
            np.ndarray: gradient of the field in x-direction (partial derivative)
        """
        grad = np.zeros_like(field)
        grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
        return grad
    
    @staticmethod
    def gradient_y(field: np.ndarray, dy: float) -> np.ndarray:
        """Compute gradient in y-direction using central differences

        Args:
            field (np.ndarray): field to differentiate
            dy (float): grid spacing in y-direction

        Returns:
            np.ndarray: gradient of the field in y-direction (partial derivative)
        """
        grad = np.zeros_like(field)
        grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dy)
        return grad
    
    @staticmethod
    def divergence(u_field: np.ndarray, v_field: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Compute divergence of velocity field on staggered grid

        Args:
            u_field (np.ndarray): x component of a vector field (MAC grid)
            v_field (np.ndarray): y component of a vector field (MAC grid)
            dx (float): grid spacing in x-direction
            dy (float): grid spacing in y-direction

        Returns:
            np.ndarray: divergence of the vector field
        """
        nx, ny = u_field.shape[0] - 1, v_field.shape[1] - 1
        div = np.zeros((nx, ny))
        
        # ∇·u = ∂u/∂x + ∂v/∂y on MAC grid
        div[:, :] = (u_field[1:, :] - u_field[:-1, :]) / dx + \
                    (v_field[:, 1:] - v_field[:, :-1]) / dy
        return div
    
    @staticmethod
    def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Compute Laplacian using 5-point stencil

        Args:
            field (np.ndarray): scalar field to compute Laplacian
            dx (float): grid spacing in x-direction
            dy (float): grid spacing in y-direction

        Returns:
            np.ndarray: Laplacian of the field
        """
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = ((field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx**2 +
                           (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / dy**2)
        return lap
    
    @staticmethod
    def interpolate_v_to_u_faces(v_field: np.ndarray) -> np.ndarray:
        """Interpolate y component of a vector field to x-face locations (based on MAC grid)

        Args:
            v_field (np.ndarray): y component of a vector field (MAC grid)

        Returns:
            np.ndarray: x component of a vector field (MAC grid)
        """
        nx_u, ny_u = v_field.shape[0] + 1, v_field.shape[1] - 1
        v_on_u = np.zeros((nx_u, ny_u))
        
        # Average v from surrounding cell faces - fix the indexing
        if nx_u > 2 and ny_u > 0:
            v_on_u[1:-1, :] = 0.25 * (v_field[:-1, :-1] + v_field[:-1, 1:] + 
                                       v_field[1:, :-1] + v_field[1:, 1:])
        return v_on_u
    
    @staticmethod
    def interpolate_u_to_v_faces(u_field: np.ndarray) -> np.ndarray:
        """Interpolate x component of a vector field to y-face locations (based on MAC grid)

        Args:
            u_field (np.ndarray): x component of a vector field (MAC grid)

        Returns:
            np.ndarray: y component of a vector field (MAC grid)
        """
        nx_v, ny_v = u_field.shape[0] - 1, u_field.shape[1] + 1
        u_on_v = np.zeros((nx_v, ny_v))
        
        # Average u from surrounding cell faces - fix the indexing
        if nx_v > 0 and ny_v > 2:
            u_on_v[:, 1:-1] = 0.25 * (u_field[:-1, :-1] + u_field[1:, :-1] + 
                                       u_field[:-1, 1:] + u_field[1:, 1:])
        return u_on_v