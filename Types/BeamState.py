import numpy as np
import numpy.typing as npt

class BeamState:
    """
    Beam state at a BPM plane
    """
    def __init__(
            self, 
            bpmIdx: int, 
            vec: npt.NDArray[np.float64], 
            cov: npt.NDArray[np.float64], 
            bpmMeas: float,
            bpmErr: float,
            bpmShift: float, 
            energy: float) -> None:
        """
        Constructor

        bpmIdx: BPM index (0 -> before the IP)
        vec: estimated beam vector (x, x') in the beamline frame [m, rad]
        cov: covariance of the beam vector [m^2, m*rad; m*rad, rad^2]
        bpmShift: BPM shift [m]
        energy: beam energy [J]
        """
        self.bpmIdx = bpmIdx
        self.vec = vec
        self.cov = cov
        self.bpmMeas = bpmMeas
        self.bpmErr = bpmErr
        self.bpmShift = bpmShift
        self.energy = energy

        self.bpmPrediction = self.vec[0] - self.bpmShift

    def getBpmPredictionVar(self) -> float:
        """
        Variance of the predicted BPM reading [m^2]
        """
        return self.cov[0,0]
