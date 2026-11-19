import numpy as np
import numpy.typing as npt

from Geometry.GeneralConstraints import GeneralConstraints
from Types.BeamOrbitFitResult import BeamOrbitFitResult, getAlignmentSampleCov

class BeamOrbitSim:
    """
    Class containing simualted fitted beam orbit
    """
    def __init__(
            self, 
            orbitFitX: BeamOrbitFitResult,
            orbitFitY: BeamOrbitFitResult,
            orbitFitBootstrapX: BeamOrbitFitResult,
            orbitFitBootstrapY: BeamOrbitFitResult,
            beamHitsTruth,
            referencePositionTruth, 
            referenceMomentumTruth, 
            energyTruth,
            alignmentShiftTruthX: npt.NDArray[np.float64],
            alignmentShiftTruthY: npt.NDArray[np.float64],
            eventId) -> None:
        def getPlanesCov(
                covX: npt.NDArray[np.float64], 
                covY: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """
            Combine the per-plane covariances into the (x, x', y, y') covariance, 
            the planes are uncorrelated

            covX: (x, x') covariance [m, rad]
            covY: (y, y') covariance [m, rad]

            return: 4x4 (x, x', y, y') covariance [m, rad]
            """
            cov = np.zeros(shape=(4, 4), dtype=np.float64)
            for i in range(2):
                for j in range(2):
                    cov[i, j] = covX[i, j]
                    cov[2 + i, 2 + j] = covY[i, j]
            return cov

        def getPulls(
                residuals: npt.NDArray[np.float64], 
                cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """
            Pulls of the residuals with the diagonal of the covariance

            residuals: residuals
            cov: covariance of the residuals

            return: pulls, 0 for the components with zero variance
            """
            sigmas = np.sqrt(np.linalg.diagonal(cov))
            pulls = np.zeros(shape=(len(residuals)), dtype=np.float64)
            for i in range(len(residuals)):
                if (sigmas[i] > 0):
                    pulls[i] = residuals[i] / sigmas[i]
            return pulls

        def getChi2(
                residuals: npt.NDArray[np.float64], 
                cov: npt.NDArray[np.float64]) -> float:
            """
            chi2 of the residuals with the covariance, 
            the pseudo-inverse is used for the singular covariances (ndf = rank)

            residuals: residuals
            cov: covariance of the residuals

            return: chi2
            """
            return float(residuals @ np.linalg.pinv(cov) @ residuals)

        nStates = len(orbitFitX.beamStates)

        self.bpmMeas = np.zeros(shape=(nStates, 2), dtype=np.float64)
        self.beamEst = np.zeros(shape=(nStates, 2), dtype=np.float64)
        self.geometryIds = np.zeros(shape=(nStates), dtype=np.float64)
        self.bpmCovs = np.zeros(shape=(nStates, 2, 2), dtype=np.float64)
        self.residualsEst = np.zeros(shape=(nStates, 2), dtype=np.float64)
        self.pullsEst = np.zeros(shape=(nStates, 2), dtype=np.float64)

        self.beamHitsTruth = np.zeros(shape=(nStates, 2), dtype=np.float64)
        self.residualsTruth = np.zeros(shape=(nStates, 2), dtype=np.float64)
        self.pullsTruth = np.zeros(shape=(nStates, 2), dtype=np.float64)
        for i in range(len(orbitFitX.beamStates)):
            stateX = orbitFitX.beamStates[i]
            stateY = orbitFitY.beamStates[i]

            # Get BPM measurements
            self.bpmMeas[i] = np.array([stateX.bpmMeas, stateY.bpmMeas])

            # Get BPM orbit estimates
            self.beamEst[i] = np.array([stateX.bpmPrediction, stateY.bpmPrediction])

            # Get BPM IDs
            self.geometryIds[i] = stateX.bpmIdx

            # Get BPM measurement covariance
            self.bpmCovs[i] = np.array([[stateX.bpmErr**2, 0], [0, stateY.bpmErr**2]])

            # Calculate residuals
            self.residualsEst[i] = self.bpmMeas[i] - self.beamEst[i]

            # Calculate pulls
            residualsEstCovTruth = np.array([[stateX.getBpmPredictionVar(), 0], [0, stateY.getBpmPredictionVar()]])
            residualsEstCov = self.bpmCovs[i] - residualsEstCovTruth
            residualsEstCovDiag = np.linalg.diagonal(residualsEstCov)
            self.pullsEst[i] = self.residualsEst[i] / np.sqrt(residualsEstCovDiag)

            # Get true beam hits
            self.beamHitsTruth[i] = beamHitsTruth[i]

            # Calculate true bpm residuals
            self.residualsTruth[i] = self.beamHitsTruth[i] - self.beamEst[i]

            # Calculate true bpm pulls
            residualsEstCovTruthDiag = np.linalg.diagonal(residualsEstCovTruth)
            self.pullsTruth[i] = self.residualsTruth[i] / np.sqrt(residualsEstCovTruthDiag)

        # Get estimated refence position
        posX = orbitFitX.referenceVec[0]
        posY = orbitFitY.referenceVec[0]
        self.referencePositionEst = np.array([posX, posY, orbitFitX.referencePos])

        # Get estimated reference momentum
        self.energyEst = orbitFitX.energyEst

        tanThetaX = np.tan(orbitFitX.referenceVec[1])
        tanThetaY = np.tan(orbitFitY.referenceVec[1])

        energyEst = (self.energyEst / GeneralConstraints.e) / 1e9
        momZ = np.sqrt((energyEst**2 - GeneralConstraints.me_mev**2) / (1 + tanThetaX**2 + tanThetaY**2)) * GeneralConstraints.e * 1e9
        momX = momZ * tanThetaX
        momY = momZ * tanThetaY
        self.referenceMomentumEst = np.array([momX, momY, momZ])

        # Orbit fit covariance at the reference (x, x', y, y') [m, rad]
        self.referenceCov = getPlanesCov(orbitFitX.referenceCov, orbitFitY.referenceCov)

        # Covariance of the beam bunch parameters at the reference (x, x', y, y') [m, rad]:
        # spot size, angular divergence and their correlation within one bunch.
        # Not to be confused with the orbit covariance at the reference plane
        # (referenceCov), the uncertainty of the reconstructed bunch centroid
        self.referenceBeamBunchCov = getPlanesCov(orbitFitX.beamBunchCov, orbitFitY.beamBunchCov)

        # Covariance at the reference due to the alignment uncertainty (x, x', y, y') [m, rad]
        # (common to all events)
        self.referenceAlignmentCov = getPlanesCov(orbitFitX.alignmentOrbitCov, orbitFitY.alignmentOrbitCov)

        # Shifts at the reference for the alignment samples (x, x', y, y') [m, rad]
        # (common to all events)
        nSamples = len(orbitFitX.alignmentSampleShifts)
        self.referenceAlignmentSampleShifts = np.zeros(shape=(nSamples, 4), dtype=np.float64)
        for k in range(nSamples):
            self.referenceAlignmentSampleShifts[k,0:2] = orbitFitX.alignmentSampleShifts[k]
            self.referenceAlignmentSampleShifts[k,2:4] = orbitFitY.alignmentSampleShifts[k]

        # Covariance at the reference from the alignment samples (x, x', y, y') [m, rad]
        # (common to all events)
        self.referenceAlignmentSampleCov = getPlanesCov(orbitFitX.alignmentSampleCov, orbitFitY.alignmentSampleCov)

        # Get energy estimate
        self.energyEst = orbitFitX.energyEst

        # Get event ID
        self.eventId = eventId

        # Get true BPM hits
        self.beamHitsTruth = beamHitsTruth

        # Get true origin position
        self.referencePositionTruth = referencePositionTruth
        
        # Get true origin momentum
        self.referenceMomentumTruth = referenceMomentumTruth

        # Get true energy
        self.energyTruth = energyTruth

        # --------------------------------------------------------------
        # Validation at the reference (x, x', y, y') [m, rad], 
        # the residuals are truth - estimate. 
        # The truth versions use the true orbit and the true alignment, 
        # the data versions (Est) use only the reconstructed quantities.

        # True (x, x', y, y') at the reference [m, rad]
        self.referenceVecTruth = np.array([
            referencePositionTruth[0], 
            np.arctan(referenceMomentumTruth[0] / referenceMomentumTruth[2]), 
            referencePositionTruth[1], 
            np.arctan(referenceMomentumTruth[1] / referenceMomentumTruth[2])], dtype=np.float64)

        # Estimated (x, x', y, y') at the reference [m, rad]
        referenceVecEst = np.array([
            orbitFitX.referenceVec[0], 
            orbitFitX.referenceVec[1], 
            orbitFitY.referenceVec[0], 
            orbitFitY.referenceVec[1]], dtype=np.float64)

        # Orbit shift at the reference from the estimated to the true alignment, 
        # S (alignmentTruth - alignment) [m, rad]
        self.referenceAlignmentShiftTruth = np.zeros(shape=(4), dtype=np.float64)
        self.referenceAlignmentShiftTruth[0:2] = alignmentShiftTruthX
        self.referenceAlignmentShiftTruth[2:4] = alignmentShiftTruthY

        # Fit: residuals with respect to the true orbit with the 
        # alignment shift removed, covariance referenceCov, ndf 4
        fitResidualsTruth = self.referenceVecTruth - referenceVecEst - self.referenceAlignmentShiftTruth
        self.referenceFitPullsTruth = getPulls(fitResidualsTruth, self.referenceCov)
        self.referenceFitChi2Truth = getChi2(fitResidualsTruth, self.referenceCov)

        # Fit, data: chi2 of the BPM residuals of both planes, ndf 4
        self.referenceFitChi2Est = orbitFitX.chi2 + orbitFitY.chi2

        # Fit, data: residuals of the estimated orbit with respect to the refit 
        # of the measurements generated from the estimated orbit with the BPM 
        # resolutions (parametric bootstrap), covariance referenceCov, ndf 4
        referenceVecBootstrap = np.array([
            orbitFitBootstrapX.referenceVec[0], 
            orbitFitBootstrapX.referenceVec[1], 
            orbitFitBootstrapY.referenceVec[0], 
            orbitFitBootstrapY.referenceVec[1]], dtype=np.float64)
        fitResidualsBootstrap = referenceVecEst - referenceVecBootstrap
        self.referenceFitBootstrapPullsEst = getPulls(fitResidualsBootstrap, self.referenceCov)
        self.referenceFitBootstrapChi2Est = getChi2(fitResidualsBootstrap, self.referenceCov)

        # Alignment: true alignment shift, covariance referenceAlignmentCov
        self.referenceAlignmentPullsTruth = getPulls(self.referenceAlignmentShiftTruth, self.referenceAlignmentCov)
        self.referenceAlignmentChi2Truth = getChi2(self.referenceAlignmentShiftTruth, self.referenceAlignmentCov)

        # Alignment, data: alignment sample shifts, covariance referenceAlignmentCov
        nSamples = len(self.referenceAlignmentSampleShifts)
        self.referenceAlignmentSamplePullsEst = np.zeros(shape=(nSamples, 4), dtype=np.float64)
        self.referenceAlignmentSampleChi2Est = np.zeros(shape=(nSamples), dtype=np.float64)
        for k in range(nSamples):
            self.referenceAlignmentSamplePullsEst[k] = getPulls(self.referenceAlignmentSampleShifts[k], self.referenceAlignmentCov)
            self.referenceAlignmentSampleChi2Est[k] = getChi2(self.referenceAlignmentSampleShifts[k], self.referenceAlignmentCov)

        # Alignment samples: true alignment shift, covariance referenceAlignmentSampleCov
        self.referenceAlignmentSampleCovPullsTruth = getPulls(self.referenceAlignmentShiftTruth, self.referenceAlignmentSampleCov)

        # Alignment samples, data: alignment sample shifts, covariance of the other 
        # samples (leave-one-out, getAlignmentSampleCov per plane), defined for at least 3 samples
        self.referenceAlignmentSampleCovPullsEst = np.zeros(shape=(nSamples, 4), dtype=np.float64)
        if (nSamples >= 3):
            for k in range(nSamples):
                otherShifts = np.delete(self.referenceAlignmentSampleShifts, k, axis=0)
                otherCov = getPlanesCov(
                        getAlignmentSampleCov(otherShifts[:,0:2], orbitFitX.alignmentSampleRmsThreshold), 
                        getAlignmentSampleCov(otherShifts[:,2:4], orbitFitY.alignmentSampleRmsThreshold))
                self.referenceAlignmentSampleCovPullsEst[k] = getPulls(self.referenceAlignmentSampleShifts[k], otherCov)

        # Total: residuals with respect to the true orbit, 
        # covariance referenceCov + referenceAlignmentCov, ndf 4
        totalResidualsTruth = self.referenceVecTruth - referenceVecEst
        self.referenceTotalPullsTruth = getPulls(totalResidualsTruth, self.referenceCov + self.referenceAlignmentCov)
        self.referenceTotalChi2Truth = getChi2(totalResidualsTruth, self.referenceCov + self.referenceAlignmentCov)
