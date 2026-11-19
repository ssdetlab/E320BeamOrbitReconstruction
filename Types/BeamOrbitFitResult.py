import numpy as np
import numpy.typing as npt

from Types.BeamState import BeamState

def getAlignmentSampleCov(shifts: npt.NDArray[np.float64], rmsThreshold: int) -> npt.NDArray[np.float64]:
    """
    Diagonal covariance of the orbit shifts for the alignment samples:
    per component, the RMS of the shifts about the current alignment 
    (shift = 0) for more than rmsThreshold samples, 
    the half-range of the shifts otherwise

    shifts: shifts with respect to the current alignment, 
        one row per sample (at least one)
    rmsThreshold: number of samples above which the RMS is used

    return: diagonal covariance, diag(RMS^2) or diag(half-range^2)
    """
    if (len(shifts) > rmsThreshold):
        width = np.sqrt(np.mean(shifts**2, axis=0))
    else:
        width = 0.5 * (np.max(shifts, axis=0) - np.min(shifts, axis=0))
    return np.diag(width**2)

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
            beamBunchCov: npt.NDArray[np.float64], 
            alignmentOrbitCov: npt.NDArray[np.float64], 
            alignmentSampleShifts: npt.NDArray[np.float64], 
            alignmentSampleCov: npt.NDArray[np.float64], 
            alignmentSampleRmsThreshold: int) -> None:
        self.beamStates = beamStates
        self.referenceVec = referenceVec
        self.referenceCov = referenceCov
        self.referencePos = referencePos
        self.energyEst = energyEst
        self.chi2 = chi2

        # Covariance of the beam bunch parameters at the reference (x, x') [m, rad]:
        # spot size, angular divergence and their correlation within one bunch.
        # Not to be confused with the orbit covariance at the reference plane
        # (referenceCov), the uncertainty of the reconstructed bunch centroid
        self.beamBunchCov = beamBunchCov

        # (x, x') covariance at the reference due to the alignment uncertainty [m, rad]
        self.alignmentOrbitCov = alignmentOrbitCov

        # (x, x') shifts at the reference for the alignment samples [m, rad]
        self.alignmentSampleShifts = alignmentSampleShifts

        # (x, x') covariance at the reference from the alignment samples, 
        # diag(RMS^2) or diag(half-range^2), see getAlignmentSampleCov [m, rad]
        self.alignmentSampleCov = alignmentSampleCov

        # Number of alignment samples above which the RMS is used
        self.alignmentSampleRmsThreshold = alignmentSampleRmsThreshold
