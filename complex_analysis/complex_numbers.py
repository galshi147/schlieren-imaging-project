import numpy as np
PI = np.pi

class Cmplxcar:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_cmplx(cls, other):
        if isinstance(other, Cmplxcar):
            x = other.real()
            y = other.imag()
            return cls(x, y)
        else:
            raise TypeError("Expected an instance of Cmplxcar")

    def __add__(self, other):
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
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def __mul__(self, other):
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
        if self.y > 0:
            return f"{self.x} + i{self.y}"
        elif self.y == 0:
            return f"{self.x}"
        elif self.y < 0:
            return f"{self.x} - i{-self.y}"

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, Cmplxcar):
            return self.x == other.x and self.y == other.y
        else:
            return False


    def is_zero(self):
        return (self.x == 0 and self.y == 0) or (self.x == 0.0 and self.y == 0.0)

    def conj(self):
        return Cmplxcar(self.x, -self.y)

    def real(self):
        return self.x

    def imag(self):
        return self.y


    def angle(self):
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
        return Cmplxrad(abs(self), self.angle())


class Cmplxrad:
    """
    :param t: must be in [0, 2*PI)
    """
    def __init__(self, r, t):
        self.r = r
        self.t = t % (2 * PI)

    def __abs__(self):
        return np.sqrt(self.r)

    def tocar(self):
        r, t = self.r, self.t
        return Cmplxcar(r * np.cos(t), r * np.sin(t))

    def __add__(self, other):
        new = self.tocar()
        return (new + other).torad()

    def __sub__(self, other):
        new = self.tocar()
        return (new - other).torad()

    def __mul__(self, other):
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
        if self.r == 0:
            return "0"
        else:
            return f"{self.r} * e^(i {self.t}"

    __repr__ = __str__

    def is_zero(self):
        return self.r == 0 or self.r == 0.0

    def conj(self):
        return Cmplxrad(self.r, 2 * PI - self.t)

    def __eq__(self, other):
        if isinstance(other, Cmplxrad):
            return self.r == other.r and self.t == other.t
        if isinstance(other, Cmplxcar):
            return self.r == abs(other) and self.t == other.angle()
        else:
            return False

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Cmplxrad(self.r ** other, (self.t * other) % (2 * PI))
        else:
            raise TypeError(f"Can power Complex number only with int or float")

    def __neg__(self):
        return self * (-1)

    def get_r(self):
        return self.r

    def set_new_radius(self, new_r):
        self.r = new_r

def generate_cmplxcar_arr(x_arr, y_arr):
    create_arr = np.vectorize(Cmplxcar)
    return create_arr(x_arr, y_arr)


def generate_cmplxrad_arr(r_arr, t_arr):
    create_arr = np.vectorize(Cmplxrad)
    return create_arr(r_arr, t_arr)


def _extract_values(cmplxcar):
    return cmplxcar.real(), cmplxcar.imag()


def extract_x_y_from_cmplxcar_arr(cmplxcar_arr):
    extract_val_arr = np.vectorize(_extract_values)
    x_arr, y_arr = extract_val_arr(cmplxcar_arr)
    return x_arr, y_arr
