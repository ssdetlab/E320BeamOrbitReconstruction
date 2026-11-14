import numpy as np

class SimBPMMeasurement:
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
        # Measurement event ID
        self.eventId = 0
        
        # True position of the beam hit
        self.trueBeamHit = np.zeros(shape=(2), dtype=float)
        # True beam origin position
        self.originPosition = np.zeros(shape=(3), dtype=float)
        # True beam origin momentum
        self.originMomentum = np.zeros(shape=(3), dtype=float)
