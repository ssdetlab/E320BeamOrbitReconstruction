import ROOT
from ctypes import c_double, c_uint64
from Geometry.GeneralConstraints import GeneralConstraints

import numpy as np
import numpy.typing as npt

class RootOrbitSimFitWriter:
    """
    Writer for the simulated fitted beam orbit
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

        # Predicted (x, x', y, y') covariance at the reference
        self.referenceCov = ROOT.TMatrixD(4, 4)

        # (x, x', y, y') beam covariance at the reference within one bunch
        self.referenceBeamCov = ROOT.TMatrixD(4, 4)

        # (x, x', y, y') measurement covariance at the reference
        self.referenceMeasurementCov = ROOT.TMatrixD(4, 4)

        # (x, x', y, y') covariance at the reference due to the alignment uncertainty
        self.referenceAlignmentCov = ROOT.TMatrixD(4, 4)

        # (x, x', y, y') shifts at the reference for the alignment samples
        self.referenceAlignmentSampleShifts = ROOT.std.vector("TVectorD")()

        # (x, x', y, y') covariance at the reference from the alignment samples
        self.referenceAlignmentSampleCov = ROOT.TMatrixD(4, 4)

        # Event ID
        self.eventId = c_uint64(0)

        # Magnets
        self.quad1Grad = c_double(0)
        self.quad2Grad = c_double(0)
        self.quad3Grad = c_double(0)
        self.steerXCOR = c_double(0)

        # --------------------------------------------------------------
        # True hits
        self.beamHitsGlobalTruth = ROOT.std.vector("TVector3")()
        self.beamHitsLocalTruth = ROOT.std.vector("TVector2")()

        # Residuals with respect to the true hits
        self.residualsTruth = ROOT.std.vector("TVector2")()

        # Pulls with respect to the true hits
        self.pullsTruth = ROOT.std.vector("TVector2")()

        # True momentum at the reference
        self.referenceMomentumTruth = ROOT.TLorentzVector()

        # True vertex at the reference
        self.referencePositionTruth = ROOT.TVector3()

        # (x, x', y, y') orbit shift at the reference from the estimated to the true alignment
        self.referenceAlignmentShiftTruth = ROOT.TVectorD(4)

        # (x, x', y, y') fit pulls at the reference and their chi2
        self.referenceFitPullsTruth = ROOT.TVectorD(4)
        self.referenceFitChi2Truth = c_double(0)
        self.referenceFitChi2Est = c_double(0)

        # (x, x', y, y') alignment pulls at the reference and their chi2
        self.referenceAlignmentPullsTruth = ROOT.TVectorD(4)
        self.referenceAlignmentChi2Truth = c_double(0)
        self.referenceAlignmentSamplePullsEst = ROOT.std.vector("TVectorD")()
        self.referenceAlignmentSampleChi2Est = ROOT.std.vector("double")()

        # (x, x', y, y') alignment sample covariance pulls at the reference
        self.referenceAlignmentSampleCovPullsTruth = ROOT.TVectorD(4)
        self.referenceAlignmentSampleCovPullsEst = ROOT.std.vector("TVectorD")()

        # (x, x', y, y') total pulls at the reference and their chi2
        self.referenceTotalPullsTruth = ROOT.TVectorD(4)
        self.referenceTotalChi2Truth = c_double(0)

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

        # Predicted (x, x', y, y') covariance at the reference
        self.tree.Branch("referenceCov", self.referenceCov, bufSize, splitLvl)

        # (x, x', y, y') beam covariance at the reference within one bunch
        self.tree.Branch("referenceBeamCov", self.referenceBeamCov, bufSize, splitLvl)

        # (x, x', y, y') measurement covariance at the reference
        self.tree.Branch("referenceMeasurementCov", self.referenceMeasurementCov, bufSize, splitLvl)

        # (x, x', y, y') covariance at the reference due to the alignment uncertainty
        self.tree.Branch("referenceAlignmentCov", self.referenceAlignmentCov, bufSize, splitLvl)

        # (x, x', y, y') shifts at the reference for the alignment samples
        self.tree.Branch("referenceAlignmentSampleShifts", self.referenceAlignmentSampleShifts, bufSize, splitLvl)

        # (x, x', y, y') covariance at the reference from the alignment samples
        self.tree.Branch("referenceAlignmentSampleCov", self.referenceAlignmentSampleCov, bufSize, splitLvl)

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

        # --------------------------------------------------------------
        # True hits
        self.tree.Branch("beamHitsGlobalTruth", self.beamHitsGlobalTruth, bufSize, splitLvl)
        self.tree.Branch("beamHitsLocalTruth", self.beamHitsLocalTruth, bufSize, splitLvl)

        # Residuals with respect to the true hits
        self.tree.Branch("residualsTruth", self.residualsTruth, bufSize, splitLvl)
        
        # Pulls with respect to the true hits
        self.tree.Branch("pullsTruth", self.pullsTruth, bufSize, splitLvl)

        # True momentum at the reference
        self.tree.Branch("referenceMomentumTruth", self.referenceMomentumTruth, bufSize, splitLvl)

        # True vertex at the reference
        self.tree.Branch("referencePositionTruth", self.referencePositionTruth, bufSize, splitLvl)

        # (x, x', y, y') orbit shift at the reference from the estimated to the true alignment
        self.tree.Branch("referenceAlignmentShiftTruth", self.referenceAlignmentShiftTruth, bufSize, splitLvl)

        # (x, x', y, y') fit pulls at the reference and their chi2
        self.tree.Branch("referenceFitPullsTruth", self.referenceFitPullsTruth, bufSize, splitLvl)
        self.tree.Branch("referenceFitChi2Truth", self.referenceFitChi2Truth, "referenceFitChi2Truth/D")
        self.tree.Branch("referenceFitChi2Est", self.referenceFitChi2Est, "referenceFitChi2Est/D")

        # (x, x', y, y') alignment pulls at the reference and their chi2
        self.tree.Branch("referenceAlignmentPullsTruth", self.referenceAlignmentPullsTruth, bufSize, splitLvl)
        self.tree.Branch("referenceAlignmentChi2Truth", self.referenceAlignmentChi2Truth, "referenceAlignmentChi2Truth/D")
        self.tree.Branch("referenceAlignmentSamplePullsEst", self.referenceAlignmentSamplePullsEst, bufSize, splitLvl)
        self.tree.Branch("referenceAlignmentSampleChi2Est", self.referenceAlignmentSampleChi2Est, bufSize, splitLvl)

        # (x, x', y, y') alignment sample covariance pulls at the reference
        self.tree.Branch("referenceAlignmentSampleCovPullsTruth", self.referenceAlignmentSampleCovPullsTruth, bufSize, splitLvl)
        self.tree.Branch("referenceAlignmentSampleCovPullsEst", self.referenceAlignmentSampleCovPullsEst, bufSize, splitLvl)

        # (x, x', y, y') total pulls at the reference and their chi2
        self.tree.Branch("referenceTotalPullsTruth", self.referenceTotalPullsTruth, bufSize, splitLvl)
        self.tree.Branch("referenceTotalChi2Truth", self.referenceTotalChi2Truth, "referenceTotalChi2Truth/D")

    def __del__(self):
        self.file.Write()
        self.file.Close()

    def write(self, fittedOrbit, magnets):
        def getCovMmRad(cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """
            Convert a (x, x', y, y') covariance to the output units

            cov: 4x4 covariance [m, rad]

            return: 4x4 covariance [mm, rad]
            """
            covScale = np.zeros(shape=(4), dtype=float)
            covScale[0] = 1e3 # x
            covScale[1] = 1.0 # x'
            covScale[2] = 1e3 # y
            covScale[3] = 1.0 # y'
            covMmRad = np.zeros(shape=(4, 4), dtype=float)
            for i in range(4):
                for j in range(4):
                    covMmRad[i, j] = cov[i, j] * covScale[i] * covScale[j]
            return covMmRad

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
            Set a (x, x', y, y') TVectorD of pulls

            vector: TVectorD to be set
            pulls: (x, x', y, y') pulls
            """
            for i in range(4):
                vector[i] = pulls[i]

        # --------------------------------------------------------------
        # Clear the truth containers

        # Beam hits
        self.beamHitsGlobalTruth.clear()
        self.beamHitsGlobalTruth.reserve(len(fittedOrbit.bpmMeas))

        self.beamHitsLocalTruth.clear()
        self.beamHitsLocalTruth.reserve(len(fittedOrbit.bpmMeas))

        # Residuals with respect to the true hits
        self.residualsTruth.clear()
        self.residualsTruth.reserve(len(fittedOrbit.bpmMeas))

        # Pulls with respect to the true hits
        self.pullsTruth.clear()
        self.pullsTruth.reserve(len(fittedOrbit.bpmMeas))

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
        # Set truth info

        # True momentum at the reference
        referenceMomentumTruth = (fittedOrbit.referenceMomentumTruth / GeneralConstraints.e) / 1e9
        energyTruth = (fittedOrbit.energyTruth / GeneralConstraints.e) / 1e9
        self.referenceMomentumTruth.SetPxPyPzE(
                referenceMomentumTruth[2], 
                referenceMomentumTruth[1], 
                -referenceMomentumTruth[0], 
                energyTruth)

        # True vertex at the reference
        referencePositionTruth = fittedOrbit.referencePositionTruth * 1e3
        self.referencePositionTruth.SetXYZ(
                referencePositionTruth[2],
                referencePositionTruth[1],
                -referencePositionTruth[0])

        # (x, x', y, y') orbit shift at the reference from the estimated to the true alignment [mm, rad]
        alignmentShiftTruth = fittedOrbit.referenceAlignmentShiftTruth
        self.referenceAlignmentShiftTruth[0] = alignmentShiftTruth[0] * 1e3 # x
        self.referenceAlignmentShiftTruth[1] = alignmentShiftTruth[1]       # x'
        self.referenceAlignmentShiftTruth[2] = alignmentShiftTruth[2] * 1e3 # y
        self.referenceAlignmentShiftTruth[3] = alignmentShiftTruth[3]       # y'

        # (x, x', y, y') fit pulls at the reference and their chi2
        setPulls(self.referenceFitPullsTruth, fittedOrbit.referenceFitPullsTruth)
        self.referenceFitChi2Truth.value = fittedOrbit.referenceFitChi2Truth
        self.referenceFitChi2Est.value = fittedOrbit.referenceFitChi2Est

        # (x, x', y, y') alignment pulls at the reference and their chi2
        setPulls(self.referenceAlignmentPullsTruth, fittedOrbit.referenceAlignmentPullsTruth)
        self.referenceAlignmentChi2Truth.value = fittedOrbit.referenceAlignmentChi2Truth

        self.referenceAlignmentSamplePullsEst.clear()
        self.referenceAlignmentSampleChi2Est.clear()
        for k in range(len(fittedOrbit.referenceAlignmentSamplePullsEst)):
            samplePulls = ROOT.TVectorD(4)
            setPulls(samplePulls, fittedOrbit.referenceAlignmentSamplePullsEst[k])
            self.referenceAlignmentSamplePullsEst.push_back(samplePulls)
            self.referenceAlignmentSampleChi2Est.push_back(fittedOrbit.referenceAlignmentSampleChi2Est[k])

        # (x, x', y, y') alignment sample covariance pulls at the reference
        setPulls(self.referenceAlignmentSampleCovPullsTruth, fittedOrbit.referenceAlignmentSampleCovPullsTruth)

        self.referenceAlignmentSampleCovPullsEst.clear()
        for samplePullsEst in fittedOrbit.referenceAlignmentSampleCovPullsEst:
            samplePulls = ROOT.TVectorD(4)
            setPulls(samplePulls, samplePullsEst)
            self.referenceAlignmentSampleCovPullsEst.push_back(samplePulls)

        # (x, x', y, y') total pulls at the reference and their chi2
        setPulls(self.referenceTotalPullsTruth, fittedOrbit.referenceTotalPullsTruth)
        self.referenceTotalChi2Truth.value = fittedOrbit.referenceTotalChi2Truth

        # --------------------------------------------------------------
        # Set observable info

        # Predicted position at the reference
        referencePositionEst = fittedOrbit.referencePositionEst * 1e3
        self.referencePositionEst.SetXYZ(
                referencePositionEst[2],
                referencePositionEst[1],
                -referencePositionEst[0])

        # Predicted (x, x', y, y') covariance at the reference [mm, rad]
        referenceCov = getCovMmRad(fittedOrbit.referenceCov)

        # Reference position pull
        referencePositionErrs = np.zeros(shape=(2), dtype=float)
        referencePositionErrs[0] = np.sqrt(referenceCov[0, 0]) # x
        referencePositionErrs[1] = np.sqrt(referenceCov[2, 2]) # y
        referencePositionPullsEst = (referencePositionTruth[:-1] - referencePositionEst[:-1]) / referencePositionErrs
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
        referenceAngleErrs[0] = np.sqrt(referenceCov[1, 1]) # x'
        referenceAngleErrs[1] = np.sqrt(referenceCov[3, 3]) # y'
        referenceAngleResiduals = np.array([self.referenceMomentumTruth.Theta() - self.referenceMomentumEst.Theta(), self.referenceMomentumTruth.Phi() - self.referenceMomentumEst.Phi()])
        referenceAnglePullsEst = referenceAngleResiduals / referenceAngleErrs
        self.referenceAnglePullsEst.Set(
                referenceAnglePullsEst[0],
                referenceAnglePullsEst[1])

        # (x, x', y, y') covariance at the reference [mm, rad]
        setCov(self.referenceCov, referenceCov)

        # (x, x', y, y') beam covariance at the reference within one bunch [mm, rad]
        setCov(self.referenceBeamCov, getCovMmRad(fittedOrbit.referenceBeamCov))

        # (x, x', y, y') measurement covariance at the reference [mm, rad]
        setCov(self.referenceMeasurementCov, getCovMmRad(fittedOrbit.referenceMeasurementCov))

        # (x, x', y, y') covariance at the reference due to the alignment uncertainty [mm, rad]
        setCov(self.referenceAlignmentCov, getCovMmRad(fittedOrbit.referenceAlignmentCov))

        # (x, x', y, y') shifts at the reference for the alignment samples [mm, rad]
        self.referenceAlignmentSampleShifts.clear()
        for shift in fittedOrbit.referenceAlignmentSampleShifts:
            sampleShift = ROOT.TVectorD(4)
            sampleShift[0] = shift[0] * 1e3 # x
            sampleShift[1] = shift[1]       # x'
            sampleShift[2] = shift[2] * 1e3 # y
            sampleShift[3] = shift[3]       # y'
            self.referenceAlignmentSampleShifts.push_back(sampleShift)

        # (x, x', y, y') covariance at the reference from the alignment samples [mm, rad]
        setCov(self.referenceAlignmentSampleCov, getCovMmRad(fittedOrbit.referenceAlignmentSampleCov))

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

            # --------------------------------------------------------------
            # True hits
            trueBeamHitLocal = ROOT.TVector2(
                    fittedOrbit.beamHitsTruth[idx][1] * 1e3,
                    fittedOrbit.beamHitsTruth[idx][0] * 1e3)
            self.beamHitsLocalTruth.push_back(trueBeamHitLocal)

            trueBeamHitGlobal = ROOT.TVector3(
                    self.cfg.bpmPositions[idx][2] * 1e3,
                    trueBeamHitLocal.X() + self.cfg.bpmPositions[idx][1] * 1e3,
                    -(trueBeamHitLocal.Y() + self.cfg.bpmPositions[idx][0] * 1e3))
            self.beamHitsGlobalTruth.push_back(trueBeamHitGlobal)

            residualTruth = ROOT.TVector2(
                        fittedOrbit.residualsTruth[idx][1] * 1e3,
                        fittedOrbit.residualsTruth[idx][0] * 1e3)
            self.residualsTruth.push_back(residualTruth)

            pullsTruth = ROOT.TVector2(
                        fittedOrbit.pullsTruth[idx][1],
                        fittedOrbit.pullsTruth[idx][0])
            self.pullsTruth.push_back(pullsTruth)

        self.tree.Fill()
