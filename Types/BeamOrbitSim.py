import numpy as np

from Geometry.GeneralConstraints import GeneralConstraints
from Types.BeamOrbitFitResult import BeamOrbitFitResult

class BeamOrbitSim:
    """
    Class containing simualted fitted beam orbit
    """
    def __init__(
            self, 
            orbitFitX: BeamOrbitFitResult,
            orbitFitY: BeamOrbitFitResult,
            beamHitsTruth,
            referencePositionTruth, 
            referenceMomentumTruth, 
            energyTruth,
            eventId) -> None:
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
            residualsEstCovTruth = np.array([[stateX.cov[0,0], 0], [0, stateY.cov[0, 0]]])
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

        # Get reference position covariance
        self.referencePositionCov = np.array([[orbitFitX.referenceCov[0][0], 0], [0, orbitFitY.referenceCov[0][0]]], dtype=np.float64)

        # Get angle estimate covariance at the reference
        self.referenceAngleCov = np.array([[orbitFitX.referenceCov[1][1], 0], [0, orbitFitY.referenceCov[1][1]]], dtype=np.float64)

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
