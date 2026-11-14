import ROOT
from ctypes import c_double, c_uint64
from Geometry.GeneralConstraints import GeneralConstraints

import numpy as np

class RootOrbitFitWriter:
    """
    Writer for the fitted beam orbit
    """
    class Config:
        """
        Nested configuration struct
        """
        # Name of the output tree
        treeName = ""
        # The names of the output file
        filePath = ""
        # BPM shifts
        bpmPositions = np.zeros(shape=(4, 3), dtype=float)

    def __init__(self, cfg: Config) -> None:
        """
        Constructor

        cfg: configuration
        """
        self.cfg = cfg

        if not self.cfg.filePath:
            raise RuntimeError("Missing file name")
        if not self.cfg.treeName:
            raise RuntimeError("Missing tree name")

        self.file = ROOT.TFile(self.cfg.filePath, "RECREATE")
        self.tree = ROOT.TTree(self.cfg.treeName, self.cfg.treeName)

        # -------------------------------------------------------------
        # Members

        # Measurement hits
        self.bpmHitsGlobal = ROOT.std.vector("TVector3")()
        self.bpmHitsLocal = ROOT.std.vector("TVector2")()

        # Covariances of the beam hits
        self.beamHitCovs = ROOT.std.vector("TMatrixD")()

        # Geometry ids of the beam hits
        self.geometryIds = ROOT.std.vector("unsigned int")()

        # Estimated beam hits
        self.beamHitsGlobalEst = ROOT.std.vector("TVector3")()
        self.beamHitsLocalEst = ROOT.std.vector("TVector2")()

        # Residuals with respect to the measurements
        self.residualsEst = ROOT.std.vector("TVector2")()

        # Pulls with respect to the measurements
        self.pullsEst = ROOT.std.vector("TVector2")()

        # Predicted vertex at the reference
        self.referencePositionEst = ROOT.TVector3()

        # Predicted momentum at the reference
        self.referenceMomentumEst = ROOT.TLorentzVector()

        # Estimated non-centered vertex pulls at the reference
        self.referencePositionPullsEst = ROOT.TVector3()

        # Estimated non-centered momentum pulls at the reference
        self.referenceAnglePullsEst = ROOT.TVector2()

        # Predicted vertex covariance at the reference
        self.referencePositionCov = ROOT.TMatrixD(2, 2)

        # Predicted angle covariance at the reference
        self.referenceAngleCov = ROOT.TMatrixD(2, 2)

        # Event ID
        self.eventId = c_uint64(0)

        # Magnets
        self.quad1Grad = c_double(0)
        self.quad2Grad = c_double(0)
        self.quad3Grad = c_double(0)
        self.steerXCOR = c_double(0)

        # -------------------------------------------------------------
        # Branches

        bufSize = 32000
        splitLvl = 0

        # Measurement hits
        self.tree.Branch("bpmHitsGlobal", self.bpmHitsGlobal, bufSize, splitLvl)
        self.tree.Branch("bpmHitsLocal", self.bpmHitsLocal, bufSize, splitLvl)

        # Covariances of the beam hits
        self.tree.Branch("beamHitsCov", self.beamHitCovs, bufSize, splitLvl)

        # Geometry ids of the beam hits
        self.tree.Branch("geometryIds", self.geometryIds, bufSize, splitLvl)

        # Estimated beam hits
        self.tree.Branch("beamHitsGlobalEst", self.beamHitsGlobalEst, bufSize, splitLvl)
        self.tree.Branch("beamHitsLocalEst", self.beamHitsLocalEst, bufSize, splitLvl)

        # Residuals with respect to the measurements
        self.tree.Branch("residualsEst", self.residualsEst, bufSize, splitLvl)

        # Pulls with respect to the measurements
        self.tree.Branch("pullsEst", self.pullsEst, bufSize, splitLvl)

        # Predicted vertex at the reference
        self.tree.Branch("referencePositionEst", self.referencePositionEst, bufSize, splitLvl)

        # Predicted momentum at the reference
        self.tree.Branch("referenceMomentumEst", self.referenceMomentumEst, bufSize, splitLvl)

        # Predicted vertex covariance at the reference
        self.tree.Branch("referencePositionCov", self.referencePositionCov, bufSize, splitLvl)

        # Predicted angle covariance at the reference
        self.tree.Branch("referenceAngleCov", self.referenceAngleCov, bufSize, splitLvl)

        # Predicted vertex pull at the reference
        self.tree.Branch("referencePositionPullsEst", self.referencePositionPullsEst, bufSize, splitLvl)

        # Predicted momentum pull at the reference
        self.tree.Branch("referenceAnglePullsEst", self.referenceAnglePullsEst, bufSize, splitLvl)

        # Event ID
        self.tree.Branch("eventId", self.eventId, "eventId/l")

        # Magnets
        self.tree.Branch("quad1Grad", self.quad1Grad, "quad1Grad/D")
        self.tree.Branch("quad2Grad", self.quad2Grad, "quad2Grad/D")
        self.tree.Branch("quad3Grad", self.quad3Grad, "quad3Grad/D")
        self.tree.Branch("steerXCOR", self.steerXCOR, "steerXCOR/D")

    def __del__(self):
        self.file.Write()
        self.file.Close()

    def write(self, fittedOrbit, magnets):
        # --------------------------------------------------------------
        # Clear the observable containers

        # Measurement hits
        self.bpmHitsGlobal.clear()
        self.bpmHitsGlobal.reserve(len(fittedOrbit.bpmMeas))

        self.bpmHitsLocal.clear()
        self.bpmHitsLocal.reserve(len(fittedOrbit.bpmMeas))

        # Covariances of the beam hits
        self.beamHitCovs.clear()
        self.beamHitCovs.reserve(len(fittedOrbit.bpmMeas))

        # Geometry ids of the beam hits
        self.geometryIds.clear()
        self.geometryIds.reserve(len(fittedOrbit.bpmMeas))

        # Estimated beam hits
        self.beamHitsGlobalEst.clear()
        self.beamHitsGlobalEst.reserve(len(fittedOrbit.bpmMeas))

        self.beamHitsLocalEst.clear()
        self.beamHitsLocalEst.reserve(len(fittedOrbit.bpmMeas))

        # Residuals with respect to the measurements
        self.residualsEst.clear()
        self.residualsEst.reserve(len(fittedOrbit.bpmMeas))

        # Pulls with respect to the measurements
        self.pullsEst.clear()
        self.pullsEst.reserve(len(fittedOrbit.bpmMeas))

        # --------------------------------------------------------------
        # Set observable info

        # Predicted position at the reference
        referencePositionEst = fittedOrbit.referencePositionEst * 1e3
        self.referencePositionEst.SetXYZ(
                referencePositionEst[2],
                referencePositionEst[1],
                -referencePositionEst[0])

        # Reference position pull
        referencePositionCov = fittedOrbit.referencePositionCov * 1e6
        referencePositionErrs = np.sqrt(np.linalg.diagonal(referencePositionCov))
        referencePositionPullsEst = referencePositionEst[:-1] / referencePositionErrs
        self.referencePositionPullsEst.SetXYZ(
                0,
                referencePositionPullsEst[1],
                -referencePositionPullsEst[0])

        # Predicted momentum at the reference
        referenceMomentumEst = (fittedOrbit.referenceMomentumEst / GeneralConstraints.e) / 1e9
        energyEst = (fittedOrbit.energyEst / GeneralConstraints.e) / 1e9
        self.referenceMomentumEst.SetPxPyPzE(
                referenceMomentumEst[2], 
                referenceMomentumEst[1], 
                -referenceMomentumEst[0], 
                energyEst)

        # Reference angle pull
        referenceAngleCov = fittedOrbit.referenceAngleCov
        referenceAngleErrs = np.sqrt(np.linalg.diagonal(referenceAngleCov))
        referenceAngleResiduals = np.array([np.pi / 2 - self.referenceMomentumEst.Theta(), self.referenceMomentumEst.Phi()])
        referenceAnglePullsEst = referenceAngleResiduals / referenceAngleErrs
        self.referenceAnglePullsEst.Set(
                referenceAnglePullsEst[0],
                referenceAnglePullsEst[1])

        # Vertex covariance at the reference
        for i in range(2):
            for j in range(2):
                self.referencePositionCov[i][j] = referencePositionCov[i, j]

        # Angle covariance at the reference
        for i in range(2):
            for j in range(2):
                self.referenceAngleCov[i][j] = referenceAngleCov[i, j]

        # Event ID
        self.eventId.value = fittedOrbit.eventId

        # Magnets
        self.quad1Grad.value = magnets[0]
        self.quad2Grad.value = magnets[1]
        self.quad3Grad.value = magnets[2]
        # self.steerXCOR.value = magnets[3]

        for idx in range(len(fittedOrbit.bpmMeas)):
            # Measurement hits
            bpmHitLocal = ROOT.TVector2(
                        fittedOrbit.bpmMeas[idx][1] * 1e3,
                        fittedOrbit.bpmMeas[idx][0] * 1e3)
            self.bpmHitsLocal.push_back(bpmHitLocal)

            bpmHitGlobal = ROOT.TVector3(
                    self.cfg.bpmPositions[idx][2] * 1e3,
                    bpmHitLocal.X() + self.cfg.bpmPositions[idx][1] * 1e3,
                    -(bpmHitLocal.Y() + self.cfg.bpmPositions[idx][0] * 1e3))
            self.bpmHitsGlobal.push_back(bpmHitGlobal)

            # Covariances of the beam hits
            beamHitCovData = ROOT.TArrayD(4)
            beamHitCov = ROOT.TMatrixD(2, 2)
            for i in range(2):
                for j in range(2):
                    beamHitCovData[i * 2 + j] = fittedOrbit.bpmCovs[idx][i, j] * 1e6
            beamHitCov.Use(2, 2, beamHitCovData.GetArray())
            self.beamHitCovs.push_back(beamHitCov)

            # Geometry ids of the beam hits
            self.geometryIds.push_back(int(fittedOrbit.geometryIds[idx]))

            # Estimated beam hits
            beamEstLocal = ROOT.TVector2(
                        fittedOrbit.beamEst[idx][1] * 1e3,
                        fittedOrbit.beamEst[idx][0] * 1e3)
            self.beamHitsLocalEst.push_back(beamEstLocal)

            beamEstGlobal = ROOT.TVector3(
                    self.cfg.bpmPositions[idx][2] * 1e3,
                    beamEstLocal.X() + self.cfg.bpmPositions[idx][1] * 1e3,
                    -(beamEstLocal.Y() + self.cfg.bpmPositions[idx][0] * 1e3))
            self.beamHitsGlobalEst.push_back(beamEstGlobal)

            residualEst = ROOT.TVector2(
                        fittedOrbit.residualsEst[idx][1] * 1e3,
                        fittedOrbit.residualsEst[idx][0] * 1e3)
            self.residualsEst.push_back(residualEst)

            pullsEst = ROOT.TVector2(
                        fittedOrbit.pullsEst[idx][1],
                        fittedOrbit.pullsEst[idx][0])
            self.pullsEst.push_back(pullsEst)

        self.tree.Fill()
