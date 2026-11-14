import numpy as np

class PropagationMatrix:
    """
    Implementation of a 2x2 ABCD propagation matrix with a
    constant offset
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        # Start with unit propagation matrix        
        self.ABCD = np.zeros(shape=(2,2), dtype=float) 
        self.ABCD[0,0] = 1.0
        self.ABCD[1,1] = 1.0

        # Start with zero offsets
        self.EF = np.zeros(shape=(2), dtype=float)

    def __str__(self) -> str:
        """
        Print the matrix
        """
        return  str(self.ABCD) + " " + str(self.EF)

    def __mul__(self, B):
        """
        Multiplication overload

        B: other PropagationMatrix
        """
        new = PropagationMatrix()
        new.ABCD = np.dot(self.ABCD, B.ABCD)
        new.EF = self.EF + np.dot(self.ABCD, B.EF) 
        return new

    def propagate(self, initial):
        """
        Propagate the beam through the beamline element

        initial: Nx2 numpy array [list of (x, x') vectors]
        """
        count = initial.shape[0]
        final = np.empty_like(initial)
        for i in range(count):
            final[i,:] = np.dot(self.ABCD, initial[i,:])
            final[i,:] += self.EF
        return final

