import ROOT
from ctypes import c_double, c_uint32, c_uint64
from Geometry.GeneralConstraints import GeneralConstraints

import numpy as np
import numpy.typing as npt

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

        # Predicted (local x, local y, phi, theta) covariance at the reference
        self.referenceCov = ROOT.TMatrixD(4, 4)

        # (local x, local y, phi, theta) beam covariance at the reference within one bunch
        self.referenceBeamCov = ROOT.TMatrixD(4, 4)

        # (local x, local y, phi, theta) measurement covariance at the reference
        self.referenceMeasurementCov = ROOT.TMatrixD(4, 4)

        # (local x, local y, phi, theta) covariance at the reference due to the alignment uncertainty
        self.referenceAlignmentCov = ROOT.TMatrixD(4, 4)

        # (local x, local y, phi, theta) shifts at the reference for the alignment samples
        self.referenceAlignmentSampleShifts = ROOT.std.vector("TVectorD")()

        # (local x, local y, phi, theta) covariance at the reference from the alignment samples
        self.referenceAlignmentSampleCov = ROOT.TMatrixD(4, 4)

        # (local x, local y, phi, theta) fit pulls at the reference and their chi2
        self.referenceFitChi2Est = c_double(0)
        self.referenceFitBootstrapPullsEst = ROOT.TVectorD(4)
        self.referenceFitBootstrapChi2Est = c_double(0)

        # (local x, local y, phi, theta) alignment pulls at the reference and their chi2
        self.referenceAlignmentSamplePullsEst = ROOT.std.vector("TVectorD")()
        self.referenceAlignmentSampleChi2Est = ROOT.std.vector("double")()

        # (local x, local y, phi, theta) alignment sample covariance pulls at the reference
        self.referenceAlignmentSampleCovPullsEst = ROOT.std.vector("TVectorD")()

        # Event ID
        self.eventId = c_uint64(0)

        # EUDAQ trigger number and run number of the event
        self.trgN = c_uint64(0)
        self.runNumber = c_uint32(0)

        # Magnets
        self.quad1Grad = c_double(0)
        self.quad2Grad = c_double(0)
        self.quad3Grad = c_double(0)

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

        # Predicted (local x, local y, phi, theta) covariance at the reference
        self.tree.Branch("referenceCov", self.referenceCov, bufSize, splitLvl)

        # (local x, local y, phi, theta) beam covariance at the reference within one bunch
        self.tree.Branch("referenceBeamCov", self.referenceBeamCov, bufSize, splitLvl)

        # (local x, local y, phi, theta) measurement covariance at the reference
        self.tree.Branch("referenceMeasurementCov", self.referenceMeasurementCov, bufSize, splitLvl)

        # (local x, local y, phi, theta) covariance at the reference due to the alignment uncertainty
        self.tree.Branch("referenceAlignmentCov", self.referenceAlignmentCov, bufSize, splitLvl)

        # (local x, local y, phi, theta) shifts at the reference for the alignment samples
        self.tree.Branch("referenceAlignmentSampleShifts", self.referenceAlignmentSampleShifts, bufSize, splitLvl)

        # (local x, local y, phi, theta) covariance at the reference from the alignment samples
        self.tree.Branch("referenceAlignmentSampleCov", self.referenceAlignmentSampleCov, bufSize, splitLvl)

        # (local x, local y, phi, theta) fit pulls at the reference and their chi2
        self.tree.Branch("referenceFitChi2Est", self.referenceFitChi2Est, "referenceFitChi2Est/D")
        self.tree.Branch("referenceFitBootstrapPullsEst", self.referenceFitBootstrapPullsEst, bufSize, splitLvl)
        self.tree.Branch("referenceFitBootstrapChi2Est", self.referenceFitBootstrapChi2Est, "referenceFitBootstrapChi2Est/D")

        # (local x, local y, phi, theta) alignment pulls at the reference and their chi2
        self.tree.Branch("referenceAlignmentSamplePullsEst", self.referenceAlignmentSamplePullsEst, bufSize, splitLvl)
        self.tree.Branch("referenceAlignmentSampleChi2Est", self.referenceAlignmentSampleChi2Est, bufSize, splitLvl)

        # (local x, local y, phi, theta) alignment sample covariance pulls at the reference
        self.tree.Branch("referenceAlignmentSampleCovPullsEst", self.referenceAlignmentSampleCovPullsEst, bufSize, splitLvl)

        # Predicted vertex pull at the reference
        self.tree.Branch("referencePositionPullsEst", self.referencePositionPullsEst, bufSize, splitLvl)

        # Predicted momentum pull at the reference
        self.tree.Branch("referenceAnglePullsEst", self.referenceAnglePullsEst, bufSize, splitLvl)

        # Event ID
        self.tree.Branch("eventId", self.eventId, "eventId/l")

        # EUDAQ trigger number and run number of the event
        self.tree.Branch("trgN", self.trgN, "trgN/l")
        self.tree.Branch("runNumber", self.runNumber, "runNumber/i")

        # Magnets
        self.tree.Branch("quad1Grad", self.quad1Grad, "quad1Grad/D")
        self.tree.Branch("quad2Grad", self.quad2Grad, "quad2Grad/D")
        self.tree.Branch("quad3Grad", self.quad3Grad, "quad3Grad/D")

    def __del__(self):
        self.file.Write()
        self.file.Close()

    def write(self, fittedOrbit, magnets):
        # Local frame of the reference plane (Acts measurement convention):
        # (local x, local y, phi, theta) = (y, x, y', x') [mm, rad]
        localIdx = np.zeros(shape=(4), dtype=int)
        localIdx[0] = 2 # y
        localIdx[1] = 0 # x
        localIdx[2] = 3 # y'
        localIdx[3] = 1 # x'
        localScale = np.zeros(shape=(4), dtype=float)
        localScale[0] = 1e3 # y
        localScale[1] = 1e3 # x
        localScale[2] = 1.0 # y'
        localScale[3] = 1.0 # x'

        def getCovLocal(cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """
            Convert a (x, x', y, y') covariance to the local frame of the reference plane

            cov: 4x4 (x, x', y, y') covariance [m, rad]

            return: 4x4 (local x, local y, phi, theta) covariance [mm, rad]
            """
            covLocal = np.zeros(shape=(4, 4), dtype=float)
            for i in range(4):
                for j in range(4):
                    covLocal[i, j] = cov[localIdx[i], localIdx[j]] * localScale[i] * localScale[j]
            return covLocal

        def getVecLocal(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """
            Convert a (x, x', y, y') vector to the local frame of the reference plane

            vec: (x, x', y, y') vector [m, rad]

            return: (local x, local y, phi, theta) vector [mm, rad]
            """
            vecLocal = np.zeros(shape=(4), dtype=float)
            for i in range(4):
                vecLocal[i] = vec[localIdx[i]] * localScale[i]
            return vecLocal

        def setCov(matrix: ROOT.TMatrixD, cov: npt.NDArray[np.float64]) -> None:
            """
            Set a 4x4 TMatrixD member

            matrix: TMatrixD member to be set
            cov: 4x4 covariance in the output units
            """
            for i in range(4):
                for j in range(4):
                    matrix[i][j] = cov[i, j]

        def setPulls(vector: ROOT.TVectorD, pulls: npt.NDArray[np.float64]) -> None:
            """
            Set a (local x, local y, phi, theta) TVectorD of pulls

            vector: TVectorD to be set
            pulls: (x, x', y, y') pulls
            """
            for i in range(4):
                vector[i] = pulls[localIdx[i]]

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

        # Predicted (local x, local y, phi, theta) covariance at the reference [mm, rad]
        referenceCov = getCovLocal(fittedOrbit.referenceCov)

        # Reference position pull
        referencePositionErrs = np.zeros(shape=(2), dtype=float)
        referencePositionErrs[0] = np.sqrt(referenceCov[1, 1]) # x
        referencePositionErrs[1] = np.sqrt(referenceCov[0, 0]) # y
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
        referenceAngleErrs = np.zeros(shape=(2), dtype=float)
        referenceAngleErrs[0] = np.sqrt(referenceCov[3, 3]) # x'
        referenceAngleErrs[1] = np.sqrt(referenceCov[2, 2]) # y'
        referenceAngleResiduals = np.array([np.pi / 2 - self.referenceMomentumEst.Theta(), self.referenceMomentumEst.Phi()])
        referenceAnglePullsEst = referenceAngleResiduals / referenceAngleErrs
        self.referenceAnglePullsEst.Set(
                referenceAnglePullsEst[0],
                referenceAnglePullsEst[1])

        # (local x, local y, phi, theta) covariance at the reference [mm, rad]
        setCov(self.referenceCov, referenceCov)

        # (local x, local y, phi, theta) beam covariance at the reference within one bunch [mm, rad]
        setCov(self.referenceBeamCov, getCovLocal(fittedOrbit.referenceBeamCov))

        # (local x, local y, phi, theta) measurement covariance at the reference [mm, rad]
        setCov(self.referenceMeasurementCov, getCovLocal(fittedOrbit.referenceMeasurementCov))

        # (local x, local y, phi, theta) covariance at the reference due to the alignment uncertainty [mm, rad]
        setCov(self.referenceAlignmentCov, getCovLocal(fittedOrbit.referenceAlignmentCov))

        # (local x, local y, phi, theta) shifts at the reference for the alignment samples [mm, rad]
        self.referenceAlignmentSampleShifts.clear()
        for shift in fittedOrbit.referenceAlignmentSampleShifts:
            shiftLocal = getVecLocal(shift)
            sampleShift = ROOT.TVectorD(4)
            for i in range(4):
                sampleShift[i] = shiftLocal[i]
            self.referenceAlignmentSampleShifts.push_back(sampleShift)

        # (local x, local y, phi, theta) covariance at the reference from the alignment samples [mm, rad]
        setCov(self.referenceAlignmentSampleCov, getCovLocal(fittedOrbit.referenceAlignmentSampleCov))

        # (local x, local y, phi, theta) fit pulls at the reference and their chi2
        self.referenceFitChi2Est.value = fittedOrbit.referenceFitChi2Est
        setPulls(self.referenceFitBootstrapPullsEst, fittedOrbit.referenceFitBootstrapPullsEst)
        self.referenceFitBootstrapChi2Est.value = fittedOrbit.referenceFitBootstrapChi2Est

        # (local x, local y, phi, theta) alignment pulls at the reference and their chi2
        self.referenceAlignmentSamplePullsEst.clear()
        self.referenceAlignmentSampleChi2Est.clear()
        for k in range(len(fittedOrbit.referenceAlignmentSamplePullsEst)):
            samplePulls = ROOT.TVectorD(4)
            setPulls(samplePulls, fittedOrbit.referenceAlignmentSamplePullsEst[k])
            self.referenceAlignmentSamplePullsEst.push_back(samplePulls)
            self.referenceAlignmentSampleChi2Est.push_back(fittedOrbit.referenceAlignmentSampleChi2Est[k])

        # (local x, local y, phi, theta) alignment sample covariance pulls at the reference
        self.referenceAlignmentSampleCovPullsEst.clear()
        for samplePullsEst in fittedOrbit.referenceAlignmentSampleCovPullsEst:
            samplePulls = ROOT.TVectorD(4)
            setPulls(samplePulls, samplePullsEst)
            self.referenceAlignmentSampleCovPullsEst.push_back(samplePulls)

        # Event ID
        self.eventId.value = fittedOrbit.eventId

        # EUDAQ trigger number and run number of the event
        self.trgN.value = fittedOrbit.trgN
        self.runNumber.value = fittedOrbit.runNumber

        # Magnets
        self.quad1Grad.value = magnets[0]
        self.quad2Grad.value = magnets[1]
        self.quad3Grad.value = magnets[2]

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
            # in the local frame of the hits, (local x, local y) = (y, x)
            beamHitCovData = ROOT.TArrayD(4)
            beamHitCov = ROOT.TMatrixD(2, 2)
            for i in range(2):
                for j in range(2):
                    beamHitCovData[i * 2 + j] = fittedOrbit.bpmCovs[idx][1 - i, 1 - j] * 1e6
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
