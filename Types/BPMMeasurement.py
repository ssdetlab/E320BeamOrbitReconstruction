import numpy as np

class BPMMeasurement:
    """
    Class containing simualted measurement data
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        # Simulated measurement
        self.measurement = np.zeros(shape=(2), dtype=float)
        # Measurement resolution
        self.cov = np.zeros(shape=(2, 2), dtype=float)
        # BPM geometry ID
        self.geoId = 0
        # EUDAQ trigger ID
        self.trigN = 0
        # EUDAQ run number
        self.runNumber = 0
