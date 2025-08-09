from complex_numbers import Cmplxcar, Cmplxrad, generate_cmplxcar_arr, generate_cmplxrad_arr, extract_x_y_from_cmplxcar_arr
import numpy as np
import pandas as pd
from typing import Union

PI = np.pi

class Airfoil:
    """Represents an airfoil shape in the complex plane.
    """
    def __init__(self, arr: np.ndarray[Cmplxcar]) -> None:
        """Initializes the Airfoil with a complex Cartesian array.

        Args:
            arr (np.ndarray[Cmplxcar]): The complex Cartesian array representing the airfoil shape.
        """
        self.arr = arr

    def get_arr(self) -> np.ndarray[Cmplxcar]:
        """Gets the complex Cartesian array representing the airfoil shape.

        Returns:
            np.ndarray[Cmplxcar]: The complex Cartesian array representing the airfoil shape.
        """
        return self.arr

    def normalize(self) -> None:
        """Normalizes the airfoil shape to fit within the unit circle.
        """
        x_arr, y_arr = extract_x_y_from_cmplxcar_arr(self.arr)
        shift = min(x_arr)
        if shift >= 0:
            x_shifted = x_arr - shift
        else:
            x_shifted = x_arr + abs(shift)
        norm_factor = max(x_shifted)
        x_norm, y_norm = x_shifted / norm_factor, y_arr / norm_factor
        self.arr = generate_cmplxcar_arr(x_norm, y_norm)
    
    def get_x_y_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Gets the x and y arrays of the airfoil shape.

        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y arrays of the airfoil shape.
        """
        return extract_x_y_from_cmplxcar_arr(self.arr)


class Circle:
    """Represents a circle in the complex plane.
    """
    def __init__(self, radius: float, center: Union[tuple, Cmplxcar, Cmplxrad] = (0, 0), points_num: int = 1000) -> None:
        self.radius = radius
        self.points_num = points_num
        center_x, center_y = 0, 0
        if isinstance(center, tuple):
            center_x, center_y = center
        elif isinstance(center, Cmplxcar):
            center_x, center_y = center.real(), center.imag()
        elif isinstance(center, Cmplxrad):
            new = center.tocar()
            center_x, center_y = new.real(), new.imag()
        self.center = (center_x, center_y)
        angles = np.linspace(0, 2 * PI, self.points_num)
        radii = np.full_like(angles, self.radius)
        rad_arr = generate_cmplxrad_arr(radii, angles)
        converted_arr_to_car = np.vectorize(Cmplxrad.tocar)
        shift = Cmplxcar(center_x, center_y)
        self.arr = converted_arr_to_car(rad_arr) + shift

    def get_arr(self) -> np.ndarray[Cmplxcar]:
        """Gets the complex Cartesian array representing the circle.

        Returns:
            np.ndarray[Cmplxcar]: The complex Cartesian array representing the circle.
        """
        return self.arr

    def get_center(self) -> Union[tuple, Cmplxcar, Cmplxrad]:
        """Gets the center of the circle.

        Returns:
            Union[tuple, Cmplxcar, Cmplxrad]: The center of the circle.
        """
        return self.center
    
    def get_x_y_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Gets the x and y arrays of the circle.

        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y arrays of the circle.
        """
        return extract_x_y_from_cmplxcar_arr(self.arr)


    def apply_joukowski(self, k: float) -> Airfoil:
        """Applies the Joukowski transform to the circle.

        Args:
            k (float): The strength of the Joukowski transform.

        Returns:
            Airfoil: The transformed airfoil shape.
        """
        jouk = lambda z: z + k ** 2 * (z.conj() / abs(z)**2)
        vec_j = np.vectorize(jouk)
        return Airfoil(vec_j(self.arr))


class PrintedAirfoil:
    """Represents a printed airfoil shape.
    """
    def __init__(self, path: str) -> None:
        self.path = path
        self.df = pd.read_csv(self.path)
        x, y = self.df["X(mm)"], self.df["Y(mm)"]
        self.x, self.y = x / max(x), y / max(x)
        self.arr = generate_cmplxcar_arr(self.x, self.y)
    
    def get_arr(self) -> np.ndarray[Cmplxcar]:
        """Gets the complex Cartesian array representing the printed airfoil.

        Returns:
            np.ndarray[Cmplxcar]: The complex Cartesian array representing the printed airfoil.
        """
        return self.arr

    def get_x_y_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Gets the x and y arrays of the printed airfoil.

        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y arrays of the printed airfoil.
        """
        return self.x.to_numpy(), self.y.to_numpy()
    
    def get_num_points(self) -> int:
        """Gets the number of points in the printed airfoil.

        Returns:
            int: The number of points in the printed airfoil.
        """
        return len(self.x)
    
    def normalize(self) -> Airfoil:
        result = Airfoil(self.arr)
        result.normalize()
        return result







