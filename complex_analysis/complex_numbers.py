import numpy as np
PI = np.pi

class Cmplxcar:
    """
    A class to represent a complex number in Cartesian form: x + iy.
    """
    def __init__(self, x, y):
        """Initialize a complex number in Cartesian form.

        Args:
            x (float): The real part.
            y (float): The imaginary part.
        """
        self.x = x
        self.y = y

    @classmethod
    def from_cmplx(cls, other):
        """Create a Cmplxcar instance from another complex representation.

        Args:
            other (Cmplxcar): The complex number to convert.

        Raises:
            TypeError: Raised if `other` is not an instance of Cmplxcar.

        Returns:
            Cmplxcar: A new Cmplxcar instance with the same real and imaginary parts as `other`.
        """
        if isinstance(other, Cmplxcar):
            x = other.real()
            y = other.imag()
            return cls(x, y)
        else:
            raise TypeError("Expected an instance of Cmplxcar")

    def __add__(self, other):
        """Add two complex numbers.

        Args:
            other (Union[Cmplxcar, Cmplxrad, int, float]): The complex number to add.

        Raises:
            TypeError: If `other` is not a supported type for addition which is only `Cmplxcar`, `Cmplxrad`, or a real number.

        Returns:
            Cmplxcar: The result of the addition as a new `Cmplxcar` instance.
        """
        if isinstance(other, Cmplxcar):
            return Cmplxcar(self.x + other.x, self.y + other.y)
        elif isinstance(other, Cmplxrad):
            new = other.tocar()
            return Cmplxcar(self.x + new.x, self.y + new.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Cmplxcar(self.x + other, self.y)
        else:
            raise TypeError(f"Can't add to Complex number a {type(other)}")

    def __sub__(self, other):
        """Subtract another complex number or a real number from this complex number.

        Args:
            other (Union[Cmplxcar, Cmplxrad, int, float]): The complex number or real number to subtract.

        Raises:
            TypeError: If `other` is not a supported type for subtraction which is only `Cmplxcar`, `Cmplxrad`, or a real number.

        Returns:
            Cmplxcar: The result of the subtraction as a new `Cmplxcar` instance.
        """
        if isinstance(other, Cmplxcar):
            return Cmplxcar(self.x - other.x, self.y - other.y)
        elif isinstance(other, Cmplxrad):
            new = other.tocar()
            return Cmplxcar(self.x - new.x, self.y - new.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Cmplxcar(self.x - other, self.y)
        else:
            raise TypeError(f"Can't add to Complex number a {type(other)}")

    def __abs__(self):
        """Calculate the magnitude (absolute value) of the complex number.

        Returns:
            float: The magnitude of the complex number.
        """
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def __mul__(self, other):
        """Multiply two complex numbers.

        Args:
            other (Union[Cmplxcar, Cmplxrad, int, float]): The complex number or real number to multiply.

        Raises:
            TypeError: If `other` is not a supported type for multiplication which is only `Cmplxcar`, `Cmplxrad`, or a real number.

        Returns:
            Cmplxcar: The result of the multiplication as a new `Cmplxcar` instance.
        """
        if isinstance(other, Cmplxcar):
            return Cmplxcar(self.x * other.x - self.y * other.y,
                            self.x * other.y + self.y * other.x)
        elif isinstance(other, Cmplxrad):
            new = other.tocar()
            return Cmplxcar(self.x * new.x - self.y * new.y,
                            self.x * new.y + self.y * new.x)
        elif isinstance(other, int) or isinstance(other, float):
            return Cmplxcar(self.x * other, self.y * other)
        else:
            raise TypeError(f"Can't multiply Complex number with {type(other)}")

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide this complex number by another complex number or a real number.

        Args:
            other (Union[Cmplxcar, Cmplxrad, int, float]): The complex number or real number to divide by.

        Raises:
            ZeroDivisionError: If `other` is zero.
            ZeroDivisionError: If `other` is a zero complex number.
            ZeroDivisionError: If `other` is a zero polar coordinate.
            TypeError: If `other` is not a supported type for division which is only `Cmplxcar`, `Cmplxrad`, or a real number.

        Returns:
            Cmplxcar: The result of the division as a new `Cmplxcar` instance.
        """
        if isinstance(other, Cmplxcar):
            try:
                new = other.torad()
                new_self = self.torad()
                return (new_self / new).tocar()
            except other.is_zero():
                raise ZeroDivisionError
        elif isinstance(other, Cmplxrad):
            try:
                new_self = self.torad()
                return (new_self / other).tocar()
            except other.is_zero():
                raise ZeroDivisionError
        elif isinstance(other, int) or isinstance(other, float):
            try:
                return Cmplxcar(self.x / other, self.y / other)
            except other == 0 or other == 0.0:
                raise ZeroDivisionError
        else:
            raise TypeError(f"Can't divide Complex number by {type(other)}")

    def __str__(self):
        """Return a string representation of the complex number.

        Returns:
            str: The string representation of the complex number.
        """
        if self.y > 0:
            return f"{self.x} + i{self.y}"
        elif self.y == 0:
            return f"{self.x}"
        elif self.y < 0:
            return f"{self.x} - i{-self.y}"

    __repr__ = __str__

    def __eq__(self, other):
        """Check if two complex numbers are equal.

        Args:
            other (Cmplxcar): The complex number to compare with.

        Returns:
            bool: True if the complex numbers are equal, False otherwise.
        """
        if isinstance(other, Cmplxcar):
            return self.x == other.x and self.y == other.y
        else:
            return False


    def is_zero(self):
        """Check if the complex number is zero.

        Returns:
            bool: True if the complex number is zero, False otherwise.
        """
        return (self.x == 0 and self.y == 0) or (self.x == 0.0 and self.y == 0.0)

    def conj(self):
        """Return the complex conjugate of the complex number.

        Returns:
            Cmplxcar: The complex conjugate of the complex number.
        """
        return Cmplxcar(self.x, -self.y)

    def real(self):
        """Return the real part of the complex number.

        Returns:
            float: The real part of the complex number.
        """
        return self.x

    def imag(self):
        """Return the imaginary part of the complex number.

        Returns:
            float: The imaginary part of the complex number.
        """
        return self.y


    def angle(self):
        """Return the angle (phase) of the complex number in radians.

        Raises:
            ValueError: If the complex number is zero.

        Returns:
            float: The angle (phase) of the complex number in radians.
        """
        x, y = self.x, self.y
        if x == 0 and y == 0:
            raise ValueError("zero has no definite angle")
        elif x > 0 and y == 0:
            return 0
        elif x > 0 and y > 0:
            return np.arctan(y / x)
        elif x == 0 and y > 0:
            return PI / 2
        elif x < 0 < y:
            return np.arctan(y / x) + PI
        elif x < 0 and y == 0:
            return PI
        elif x < 0 and y < 0:
            return np.arctan(y / x) + PI
        elif x == 0 and y < 0:
            return 3 * PI / 2
        elif x > 0 > y:
            return np.arctan(y / x) + 2 * PI

    def torad(self):
        """Convert the complex number to polar form (radius and angle).

        Returns:
            Cmplxrad: The polar representation of the complex number.
        """
        return Cmplxrad(abs(self), self.angle())


class Cmplxrad:
    """Polar representation of a complex number.
    """
    def __init__(self, r, t):
        """Initialize the polar representation of a complex number.

        Args:
            r (float): The radius (magnitude) of the complex number.
            t (float): The angle (phase) of the complex number in radians. must be in [0, 2*PI)
            """
        self.r = r
        self.t = t % (2 * PI)

    def __abs__(self):
        """Return the magnitude (absolute value) of the complex number.

        Returns:
            float: The magnitude (absolute value) of the complex number.
        """
        return np.sqrt(self.r)

    def tocar(self):
        """Convert the polar representation to rectangular form (Cartesian coordinates).

        Returns:
            Cmplxcar: The rectangular representation of the complex number.
        """
        r, t = self.r, self.t
        return Cmplxcar(r * np.cos(t), r * np.sin(t))

    def __add__(self, other):
        """Add two complex numbers.

        Args:
            other (Cmplxrad or Cmplxcar): The complex number to add.

        Returns:
            Cmplxrad: The sum of the complex numbers.
        """
        new = self.tocar()
        return (new + other).torad()

    def __sub__(self, other):
        """Subtract two complex numbers.

        Args:
            other (Cmplxrad or Cmplxcar): The complex number to subtract.

        Returns:
            Cmplxrad: The difference of the complex numbers.
        """
        new = self.tocar()
        return (new - other).torad()

    def __mul__(self, other):
        """Multiply two complex numbers.

        Args:
            other (Cmplxrad or Cmplxcar): The complex number to multiply.

        Raises:
            TypeError: If `other` is not a supported type for multiplication which is only `Cmplxrad`, `Cmplxcar`, or a real number.

        Returns:
            Cmplxrad: The product of the complex numbers.
        """
        if isinstance(other, Cmplxrad):
            return Cmplxrad(self.r * other.r, (self.t + other.t) % (2 * PI))
        elif isinstance(other, Cmplxcar):
            new = other.torad()
            return Cmplxrad(self.r * new.r, (self.t + new.t) % (2 * PI))
        elif isinstance(other, int) or isinstance(other, float):
            return Cmplxrad(other * self.r, self.t)
        else:
            raise TypeError(
                f"Can't multiply Complex number with {type(other)}")

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide two complex numbers.

        Args:
            other (Union[Cmplxcar, Cmplxrad, int, float]): The complex number to divide.

        Raises:
            ZeroDivisionError: If `other` is zero.
            ZeroDivisionError: If `other` is a zero complex number.
            ZeroDivisionError: If `other` is a zero polar coordinate.
            TypeError: If `other` is not a supported type for division which is only `Cmplxrad`, `Cmplxcar`, or a real number.

        Returns:
            Cmplxrad: The quotient of the complex numbers.
        """
        if isinstance(other, Cmplxrad):
            try:
                fix = 0
                if (self.t - other.t) < 0:
                    fix = 2 * PI
                return Cmplxrad(self.r / other.r, (self.t - other.t + fix) % 2 * PI)
            except other.is_zero():
                raise ZeroDivisionError
        elif isinstance(other, Cmplxcar):
            try:
                new = other.torad()
                fix = 0
                if (self.t - new.t) < 0:
                    fix = 2 * PI
                return Cmplxrad(self.r / new.r,
                                (self.t - new.t + fix) % 2 * PI)
            except other.is_zero():
                raise ZeroDivisionError
        elif isinstance(other, int) or isinstance(other, float):
            try:
                return Cmplxrad(self.r / other, self.t)
            except other == 0 or other == 0.0:
                raise ZeroDivisionError
        else:
            raise TypeError(f"Can't divide Complex number by {type(other)}")

    def __str__(self):
        """Convert the complex number to a string representation.

        Returns:
            str: The string representation of the complex number.
        """
        if self.r == 0:
            return "0"
        else:
            return f"{self.r} * e^(i {self.t}"

    __repr__ = __str__

    def is_zero(self):
        """Check if the complex number is zero.

        Returns:
            bool: True if the complex number is zero, False otherwise.
        """
        return self.r == 0 or self.r == 0.0

    def conj(self):
        """Get the complex conjugate of the complex number.

        Returns:
            Cmplxrad: The complex conjugate of the complex number.
        """
        return Cmplxrad(self.r, 2 * PI - self.t)

    def __eq__(self, other):
        """Check if two complex numbers are equal.

        Args:
            other (Union[Cmplxcar, Cmplxrad]): The complex number to compare.

        Returns:
            bool: True if the complex numbers are equal, False otherwise.
        """
        if isinstance(other, Cmplxrad):
            return self.r == other.r and self.t == other.t
        if isinstance(other, Cmplxcar):
            return self.r == abs(other) and self.t == other.angle()
        else:
            return False

    def __pow__(self, other):
        """Raise the complex number to a power.

        Args:
            other (Union[int, float]): The exponent to raise the complex number to.

        Raises:
            TypeError: If `other` is not an int or float.

        Returns:
            Cmplxrad: The complex number raised to the power of `other`.
        """
        if isinstance(other, int) or isinstance(other, float):
            return Cmplxrad(self.r ** other, (self.t * other) % (2 * PI))
        else:
            raise TypeError(f"Can power Complex number only with int or float")

    def __neg__(self):
        """Get the additive inverse of the complex number.

        Returns:
            Cmplxrad: The additive inverse of the complex number.
        """
        return self * (-1)

    def get_r(self):
        """Get the radius (magnitude) of the complex number.

        Returns:
            float: The radius (magnitude) of the complex number.
        """
        return self.r

    def set_new_radius(self, new_r):
        """Set a new radius (magnitude) for the complex number.

        Args:
            new_r (float): The new radius (magnitude) of the complex number.
        """
        self.r = new_r

def generate_cmplxcar_arr(x_arr: np.ndarray[float], y_arr: np.ndarray[float]) -> np.ndarray[Cmplxcar]:
    """Generate an array of complex numbers in Cartesian form.

    Args:
        x_arr (np.ndarray): The x-coordinates (real part) of the complex numbers.
        y_arr (np.ndarray): The y-coordinates (imaginary part) of the complex numbers.

    Returns:
        np.ndarray: An array of complex numbers in Cartesian form.
    """
    create_arr = np.vectorize(Cmplxcar)
    return create_arr(x_arr, y_arr)


def generate_cmplxrad_arr(r_arr: np.ndarray[float], t_arr: np.ndarray[float]) -> np.ndarray[Cmplxrad]:
    """Generate an array of complex numbers in polar form.

    Args:
        r_arr (np.ndarray): The radii (magnitudes) of the complex numbers.
        t_arr (np.ndarray): The angles (phases) of the complex numbers.

    Returns:
        np.ndarray: An array of complex numbers in polar form.
    """
    create_arr = np.vectorize(Cmplxrad)
    return create_arr(r_arr, t_arr)


def _extract_values(cmplxcar: Cmplxcar) -> tuple[float, float]:
    """Extract the real and imaginary parts from a complex number in Cartesian form.

    Args:
        cmplxcar (Cmplxcar): The complex number in Cartesian form.

    Returns:
        tuple: The real and imaginary parts of the complex number. (x, y)
    """
    return cmplxcar.real(), cmplxcar.imag()


def extract_x_y_from_cmplxcar_arr(cmplxcar_arr: np.ndarray[Cmplxcar]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Extract the real and imaginary parts from an array of complex numbers in Cartesian form.

    Args:
        cmplxcar_arr (np.ndarray[Cmplxcar]): The array of complex numbers in Cartesian form.

    Returns:
        tuple: A tuple containing two arrays - the real parts (x) and the imaginary parts (y).
    """
    extract_val_arr = np.vectorize(_extract_values)
    x_arr, y_arr = extract_val_arr(cmplxcar_arr)
    return x_arr, y_arr
