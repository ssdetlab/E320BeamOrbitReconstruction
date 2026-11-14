import numpy as np
import numpy.typing as npt

from Types.BeamState import BeamState

class BeamOrbitFitResult:
    """
    Class containing simualted fitted beam orbit
    """
    def __init__(
            self, 
            beamStates: list[BeamState],
            referenceVec: npt.NDArray[np.float64],
            referenceCov: npt.NDArray[np.float64],
            referencePos: float,
            energyEst: float,
            chi2) -> None:
        self.beamStates = beamStates
        self.referenceVec = referenceVec
        self.referenceCov = referenceCov
        self.referencePos = referencePos
        self.energyEst = energyEst
        self.chi2 = chi2
