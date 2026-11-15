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
            chi2, 
            beamCov: npt.NDArray[np.float64], 
            alignmentOrbitCov: npt.NDArray[np.float64], 
            alignmentSampleShifts: npt.NDArray[np.float64], 
            alignmentSampleCov: npt.NDArray[np.float64]) -> None:
        self.beamStates = beamStates
        self.referenceVec = referenceVec
        self.referenceCov = referenceCov
        self.referencePos = referencePos
        self.energyEst = energyEst
        self.chi2 = chi2

        # Beam (x, x') covariance at the reference within one bunch [m, rad]
        self.beamCov = beamCov

        # (x, x') covariance at the reference due to the alignment uncertainty [m, rad]
        self.alignmentOrbitCov = alignmentOrbitCov

        # (x, x') shifts at the reference for the alignment samples [m, rad]
        self.alignmentSampleShifts = alignmentSampleShifts

        # (x, x') covariance at the reference from the alignment samples, 
        # diag(half-range^2) [m, rad]
        self.alignmentSampleCov = alignmentSampleCov
