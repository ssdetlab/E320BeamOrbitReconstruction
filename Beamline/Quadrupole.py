from Beamline.PropagationMatrix import PropagationMatrix
from Geometry.GeneralConstraints import GeneralConstraints

import numpy as np

class Quadrupole(PropagationMatrix):
    """
    Quad magnet base class
    """
    def __init__(self, grad: float, l: float, energy: float, offset: float, tilt: float) -> None:
        """
            grad:   [Tesla/m]
            Lquad:  [meter]
            Energy: [Joule]

        """
        super().__init__()

        self.gamma = energy / (GeneralConstraints.me_kg * GeneralConstraints.c * GeneralConstraints.c)

        beta = 1.0 / np.sqrt(1 - 1.0 / self.gamma**2)
        self.v = beta * GeneralConstraints.c

        self.freq = np.sqrt(GeneralConstraints.e * self.v * grad / (self.gamma * GeneralConstraints.me_kg))

        t = l / self.v 
        self.phase = self.freq * t

        self.offset = offset
        self.tilt = tilt

class FocusQuadrupole(Quadrupole):
    """
    Quad Magnet: focusing
    """
    def __init__(self, grad: float, l: float, energy: float, offset: float, tilt: float) -> None:
        super().__init__(grad, l, energy, offset, tilt)

        self.ABCD[0,0] = np.cos(self.phase)
        self.ABCD[0,1] = (self.v / self.freq) * np.sin(self.phase)
        self.ABCD[1,0] = -(self.freq / self.v) * np.sin(self.phase)
        self.ABCD[1,1] = self.ABCD[0,0]

        # Offset EF
        self.EF[0] = (1 - self.ABCD[0, 0]) * self.offset 
        self.EF[1] = -self.ABCD[1, 0] * self.offset 

        # Tilt EF
        self.EF[0] = self.EF[0] + self.tilt * ((1 + self.ABCD[0, 0]) * l/2 - self.ABCD[0,1]) 
        self.EF[1] = self.EF[1] + self.tilt * (1 - self.ABCD[1, 1] + self.ABCD[1,0] * l/2) 

class DefocusQuadrupole(Quadrupole):
    """
    Quad Magnet: focusing
    """
    def __init__(self, grad: float, l: float, energy: float, offset: float, tilt: float) -> None:
        super().__init__(grad, l, energy, offset, tilt)

        self.ABCD[0,0] = np.cosh(self.phase)
        self.ABCD[0,1] = (self.v / self.freq) * np.sinh(self.phase)
        self.ABCD[1,0] = (self.freq / self.v) * np.sinh(self.phase)
        self.ABCD[1,1] = self.ABCD[0,0]

        # Offset EF
        self.EF[0] = (1 - self.ABCD[0, 0]) * self.offset 
        self.EF[1] = -self.ABCD[1, 0] * self.offset 

        # Tilt EF
        self.EF[0] = self.EF[0] + self.tilt * ((1 + self.ABCD[0, 0]) * l/2 - self.ABCD[0,1]) 
        self.EF[1] = self.EF[1] + self.tilt * (1 - self.ABCD[1, 1] + self.ABCD[1,0] * l/2) 
