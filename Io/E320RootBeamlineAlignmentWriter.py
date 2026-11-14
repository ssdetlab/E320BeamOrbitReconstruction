import ROOT

import numpy as np
import numpy.typing as npt
from ctypes import c_int, c_double

class E320RootBeamlineAlignmentWriter:
    """
    Alignment parameters writer

    The alignment is written as vectors. 
    The convention is [bpm0, bpm1, bpm2, bpm3, quad1, quad2, quad3] separate for X and Y.
    """
    class Config:
        """
        Nested configuration struct
        """
        # Name of the output tree
        treeName = ""
        # Name of the output file
        filePath = ""

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

        self.file = ROOT.TFile(self.cfg.filePath, "RECREATE");
        self.tree = ROOT.TTree(self.cfg.treeName, self.cfg.treeName);

        # -------------------------------------------------------------
        # Members

        # [x, y, z] convention for the position vectors, 
        # [pitch, yaw] for quad angles

        # BPM positions [mm]
        self.bpm0Position = ROOT.TVector3()
        self.bpm1Position = ROOT.TVector3()
        self.bpm2Position = ROOT.TVector3()
        self.bpm3Position = ROOT.TVector3()

        # BPM constraints means [mm]
        self.bpm0PositionConstraintsMean = ROOT.TVector3()
        self.bpm1PositionConstraintsMean = ROOT.TVector3()
        self.bpm2PositionConstraintsMean = ROOT.TVector3()
        self.bpm3PositionConstraintsMean = ROOT.TVector3()

        # BPM constraints uncertainties [mm]
        self.bpm0PositionConstraintsSigma = ROOT.TVector3()
        self.bpm1PositionConstraintsSigma = ROOT.TVector3()
        self.bpm2PositionConstraintsSigma = ROOT.TVector3()
        self.bpm3PositionConstraintsSigma = ROOT.TVector3()

        # Quad positions [mm]
        self.quad1Position = ROOT.TVector3()
        self.quad2Position = ROOT.TVector3()
        self.quad3Position = ROOT.TVector3()

        # Quad tilts [rad]
        self.quad1Tilt = ROOT.TVector2()
        self.quad2Tilt = ROOT.TVector2()
        self.quad3Tilt = ROOT.TVector2()

        # Quad shifts constraints means [mm]
        self.quad1PositionConstraintsMean = ROOT.TVector3()
        self.quad2PositionConstraintsMean = ROOT.TVector3()
        self.quad3PositionConstraintsMean = ROOT.TVector3()

        # Quad shifts constraints uncertainties [mm]
        self.quad1PositionConstraintsSigma = ROOT.TVector3()
        self.quad2PositionConstraintsSigma = ROOT.TVector3()
        self.quad3PositionConstraintsSigma = ROOT.TVector3()

        # Quad tilts constraints means [rad]
        self.quad1TiltConstraintsMean = ROOT.TVector2()
        self.quad2TiltConstraintsMean = ROOT.TVector2()
        self.quad3TiltConstraintsMean = ROOT.TVector2()

        # Quad tilts constraints uncertainties [rad]
        self.quad1TiltConstraintsSigma = ROOT.TVector2()
        self.quad2TiltConstraintsSigma = ROOT.TVector2()
        self.quad3TiltConstraintsSigma = ROOT.TVector2()

        # BPM fit status (true == floating, false == fixed)
        # [bpm0, bpm1, bpm2, bpm3]
        self.bpmFree = ROOT.std.vector("bool")()

        # Quad shifts fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        self.quadShiftsFree = ROOT.std.vector("bool")()

        # Quad tilits fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        self.quadTiltsFree = ROOT.std.vector("bool")()

        # Alignment fit chi2
        self.alignmentLongChi2 = c_double(0)
        self.alignmentShortChi2 = c_double(0)

        # Alignment fit ndf
        self.alignmentLongNdf = c_int(0)
        self.alignmentShortNdf = c_int(0)

        # Alignment fit covariance
        self.alignmentCov = ROOT.TMatrixD(20, 20)

        # BPM uncertainties used in the alignment fit
        # in local BPM coordinates
        # [bpm0, bpm1, bpm2, bpm3]
        self.bpmHitCovs = ROOT.std.vector("TMatrixD")()

        # -------------------------------------------------------------
        # Tree branches

        bufSize = 32000
        splitLvl = 0

        # BPM positions [mm]
        self.tree.Branch("bpm0Position", self.bpm0Position, bufSize, splitLvl)
        self.tree.Branch("bpm1Position", self.bpm1Position, bufSize, splitLvl)
        self.tree.Branch("bpm2Position", self.bpm2Position, bufSize, splitLvl)
        self.tree.Branch("bpm3Position", self.bpm3Position, bufSize, splitLvl)

        # BPM constraints means [mm]
        self.tree.Branch(
                "bpm0PositionConstraintsMean", 
                self.bpm0PositionConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm1PositionConstraintsMean", 
                self.bpm1PositionConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm2PositionConstraintsMean", 
                self.bpm2PositionConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm3PositionConstraintsMean", 
                self.bpm3PositionConstraintsMean, 
                bufSize, 
                splitLvl)

        # BPM constraints uncertainties [mm]
        self.tree.Branch(
                "bpm0PositionConstraintsSigma", 
                self.bpm0PositionConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm1PositionConstraintsSigma", 
                self.bpm1PositionConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm2PositionConstraintsSigma", 
                self.bpm2PositionConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "bpm3PositionConstraintsSigma", 
                self.bpm3PositionConstraintsSigma, 
                bufSize, 
                splitLvl)

        # Quad positions [mm]
        self.tree.Branch("quad1Position", self.quad1Position, bufSize, splitLvl)
        self.tree.Branch("quad2Position", self.quad2Position, bufSize, splitLvl)
        self.tree.Branch("quad3Position", self.quad3Position, bufSize, splitLvl)

        # Quad tilts [rad]
        self.tree.Branch("quad1Tilt", self.quad1Tilt, bufSize, splitLvl)
        self.tree.Branch("quad2Tilt", self.quad2Tilt, bufSize, splitLvl)
        self.tree.Branch("quad3Tilt", self.quad3Tilt, bufSize, splitLvl)

        # Quad shifts constraints means [mm]
        self.tree.Branch(
                "quad1PositionConstraintsMean", 
                self.quad1PositionConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad2PositionConstraintsMean", 
                self.quad2PositionConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad3PositionConstraintsMean", 
                self.quad3PositionConstraintsMean, 
                bufSize, 
                splitLvl)

        # Quad shifts constraints uncertainties [mm]
        self.tree.Branch(
                "quad1PositionConstraintsSigma", 
                self.quad1PositionConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad2PositionConstraintsSigma", 
                self.quad2PositionConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad3PositionConstraintsSigma", 
                self.quad3PositionConstraintsSigma, 
                bufSize, 
                splitLvl)

        # Quad tilts constraints means [rad]
        self.tree.Branch(
                "quad1TiltConstraintsMean", 
                self.quad1TiltConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad2TiltConstraintsMean", 
                self.quad2TiltConstraintsMean, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad3TiltConstraintsMean", 
                self.quad3TiltConstraintsMean, 
                bufSize, 
                splitLvl)

        # Quad tilts constraints uncertainties [rad]
        self.tree.Branch(
                "quad1TiltConstraintsSigma", 
                self.quad1TiltConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad2TiltConstraintsSigma", 
                self.quad2TiltConstraintsSigma, 
                bufSize, 
                splitLvl)
        self.tree.Branch(
                "quad3TiltConstraintsSigma", 
                self.quad3TiltConstraintsSigma, 
                bufSize, 
                splitLvl)

        # BPM fit status (true == floating, false == fixed)
        # [bpm0, bpm1, bpm2, bpm3]
        self.tree.Branch("bpmFree", self.bpmFree, bufSize, splitLvl)

        # Quad shifts fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        self.tree.Branch("quadShiftsFree", self.quadShiftsFree, bufSize, splitLvl)

        # Quad tilits fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        self.tree.Branch("quadTiltsFree", self.quadTiltsFree, bufSize, splitLvl)

        # Alignment fit chi2
        self.tree.Branch("alignmentLongChi2", self.alignmentLongChi2, "alignmentLongChi2/D")
        self.tree.Branch("alignmentShortChi2", self.alignmentShortChi2, "alignmentShortChi2/D")

        # Alignment fit ndf
        self.tree.Branch("alignmentLongNdf", self.alignmentLongNdf, "alignmentLongNdf/I")
        self.tree.Branch("alignmentShortNdf", self.alignmentShortNdf, "alignmentShortNdf/I")

        # Alignment fit covariance
        self.tree.Branch("alignmentCov", self.alignmentCov, bufSize, splitLvl)

        # BPM uncertainties used in the alignment fit
        # in local BPM coordinates
        # [bpm0, bpm1, bpm2, bpm3]
        self.tree.Branch("bpmHitCovs", self.bpmHitCovs, bufSize, splitLvl)

    def __del__(self) -> None:
        self.file.Write()
        self.file.Close()

    def write(
            self, 
            bpmPositions: list[npt.NDArray[np.float64]], 
            bpmPositionConstraintsMeans: list[npt.NDArray[np.float64]], 
            bpmPositionConstraintsSigmas: list[npt.NDArray[np.float64]], 
            quadPositions: list[npt.NDArray[np.float64]], 
            quadTilts: list[npt.NDArray[np.float64]], 
            quadPositionConstraintsMeans: list[npt.NDArray[np.float64]], 
            quadTiltConstraintsMeans: list[npt.NDArray[np.float64]], 
            quadPositionConstraintsSigmas: list[npt.NDArray[np.float64]], 
            quadTiltConstraintsSigmas: list[npt.NDArray[np.float64]], 
            bpmFree: npt.NDArray[np.bool], 
            quadShiftsFree: npt.NDArray[np.bool], 
            quadTiltsFree: npt.NDArray[np.bool], 
            alignmentLongChi2: float,
            alignmentShortChi2: float,
            alignmentLongNdf: int,
            alignmentShortNdf: int,
            alignmentCov: npt.NDArray[np.float64], 
            bpmCovs: list[npt.NDArray[np.float64]]) -> None:
        """
        Write alignment parameters

        bpmPositions: BPM positions [bpm0, bpm1, bpm2, bpm3], [m]
        bpmPositionConstraintsMeans: BPM position constraint means [bpm0, bpm1, bpm2, bpm3], [m] 
        bpmPositionConstraintsSigmas: BPM position constraint uncertainties [bpm0, bpm1, bpm2, bpm3], [m]  
        quadPositions: quad positions [quad1, quad2, quad3], [m]  
        quadTilts: quad tilts [quad1, quad2, quad3], [rad] 
        quadPositionConstraintsMeans: quad position constraint means [quad1, quad2, quad3], [m]  
        quadTiltConstraintsMeans: quad tilt constraint means [quad1, quad2, quad3], [m]  
        quadPositionConstraintsSigmas: quad position constraint uncertainties [quad1, quad2, quad3], [m]    
        quadTiltConstraintsSigmas: quad tilt constraint uncertainties [quad1, quad2, quad3], [m]     
        bpmFree: BPMs fit status (floating == True, fixed == False) [bpm0, bpm1, bpm2, bpm3]
        quadShiftsFree: quad shifts fit status (floating == True, fixed == False) [bpm0, bpm1, bpm2, bpm3] 
        quadTiltsFree: quad tilts fit status (floating == True, fixed == False) [bpm0, bpm1, bpm2, bpm3]  
        alignmentLongChi2: alignment fit chi2 in the long direction
        alignmentShortChi2: alignment fit chi2 in the short direction
        alignmentLongNdf: alignment fit ndf in the long direction
        alignmentShortNdf: alignment fit ndf in the short direction
        alignmentCov: alignment fit covariance in the format 
            [bpm0, bpm1, bpm2, bpm3, quad1, quad2, quad3]_Short X [bpm0, bpm1, bpm2, bpm3, quad1, quad2, quad3]_Long, [m, rad]
        bpmCovs: BPM covariances used in the alignment fit in the format: 
            [bpm0_x X bpm0_y, bpm1_x X bpm1_y, bpm2_x X bpm2_y, bpm3_x X bpm3_y], [m]
        """
        # Clear the vectors
        self.bpmHitCovs.clear()
        self.bpmFree.clear()
        self.quadShiftsFree.clear()
        self.quadTiltsFree.clear()

        # BPM positions [mm]
        self.bpm0Position.SetXYZ(
                bpmPositions[0][2] * 1e3,
                bpmPositions[0][1] * 1e3,
                -bpmPositions[0][0] * 1e3)
        self.bpm1Position.SetXYZ(
                bpmPositions[1][2] * 1e3,
                bpmPositions[1][1] * 1e3,
                -bpmPositions[1][0] * 1e3)
        self.bpm2Position.SetXYZ(
                bpmPositions[2][2] * 1e3,
                bpmPositions[2][1] * 1e3,
                -bpmPositions[2][0] * 1e3)
        self.bpm3Position.SetXYZ(
                bpmPositions[3][2] * 1e3,
                bpmPositions[3][1] * 1e3,
                -bpmPositions[3][0] * 1e3)

        # BPM constraints means [mm]
        self.bpm0PositionConstraintsMean.SetXYZ(
                bpmPositionConstraintsMeans[0][2] * 1e3,
                bpmPositionConstraintsMeans[0][1] * 1e3,
                -bpmPositionConstraintsMeans[0][0] * 1e3)
        self.bpm1PositionConstraintsMean.SetXYZ(
                bpmPositionConstraintsMeans[1][2] * 1e3,
                bpmPositionConstraintsMeans[1][1] * 1e3,
                -bpmPositionConstraintsMeans[1][0] * 1e3)
        self.bpm2PositionConstraintsMean.SetXYZ(
                bpmPositionConstraintsMeans[2][2] * 1e3,
                bpmPositionConstraintsMeans[2][1] * 1e3,
                -bpmPositionConstraintsMeans[2][0] * 1e3)
        self.bpm3PositionConstraintsMean.SetXYZ(
                bpmPositionConstraintsMeans[3][2] * 1e3,
                bpmPositionConstraintsMeans[3][1] * 1e3,
                -bpmPositionConstraintsMeans[3][0] * 1e3)

        # BPM constraints uncertainties [mm]
        self.bpm0PositionConstraintsSigma.SetXYZ(
                bpmPositionConstraintsSigmas[0][2] * 1e3 if bpmPositionConstraintsSigmas[0][2] != np.inf else -1,
                bpmPositionConstraintsSigmas[0][1] * 1e3 if bpmPositionConstraintsSigmas[0][1] != np.inf else -1,
                bpmPositionConstraintsSigmas[0][0] * 1e3 if bpmPositionConstraintsSigmas[0][0] != np.inf else -1)
        self.bpm1PositionConstraintsSigma.SetXYZ(
                bpmPositionConstraintsSigmas[1][2] * 1e3 if bpmPositionConstraintsSigmas[1][2] != np.inf else -1,
                bpmPositionConstraintsSigmas[1][1] * 1e3 if bpmPositionConstraintsSigmas[1][1] != np.inf else -1,
                bpmPositionConstraintsSigmas[1][0] * 1e3 if bpmPositionConstraintsSigmas[1][0] != np.inf else -1)
        self.bpm2PositionConstraintsSigma.SetXYZ(
                bpmPositionConstraintsSigmas[2][2] * 1e3 if bpmPositionConstraintsSigmas[2][2] != np.inf else -1,
                bpmPositionConstraintsSigmas[2][1] * 1e3 if bpmPositionConstraintsSigmas[2][1] != np.inf else -1,
                bpmPositionConstraintsSigmas[2][0] * 1e3 if bpmPositionConstraintsSigmas[2][0] != np.inf else -1)
        self.bpm3PositionConstraintsSigma.SetXYZ(
                bpmPositionConstraintsSigmas[3][2] * 1e3 if bpmPositionConstraintsSigmas[3][2] != np.inf else -1,
                bpmPositionConstraintsSigmas[3][1] * 1e3 if bpmPositionConstraintsSigmas[3][1] != np.inf else -1,
                bpmPositionConstraintsSigmas[3][0] * 1e3 if bpmPositionConstraintsSigmas[3][0] != np.inf else -1)

        # Quad positions [mm]
        self.quad1Position.SetXYZ(
                quadPositions[0][2] * 1e3,
                quadPositions[0][1] * 1e3,
                -quadPositions[0][0] * 1e3)
        self.quad2Position.SetXYZ(
                quadPositions[1][2] * 1e3,
                quadPositions[1][1] * 1e3,
                -quadPositions[1][0] * 1e3)
        self.quad3Position.SetXYZ(
                quadPositions[2][2] * 1e3,
                quadPositions[2][1] * 1e3,
                -quadPositions[2][0] * 1e3)

        # Quad tilts [rad]
        self.quad1Tilt.Set(quadTilts[0][0], quadTilts[0][1])
        self.quad2Tilt.Set(quadTilts[1][0], quadTilts[1][1])
        self.quad3Tilt.Set(quadTilts[2][0], quadTilts[2][1])

        # Quad shifts constraints means [mm]
        self.quad1PositionConstraintsMean.SetXYZ(
                quadPositionConstraintsMeans[0][2] * 1e3,
                quadPositionConstraintsMeans[0][1] * 1e3,
                -quadPositionConstraintsMeans[0][0] * 1e3)
        self.quad2PositionConstraintsMean.SetXYZ(
                quadPositionConstraintsMeans[1][2] * 1e3,
                quadPositionConstraintsMeans[1][1] * 1e3,
                -quadPositionConstraintsMeans[1][0] * 1e3)
        self.quad3PositionConstraintsMean.SetXYZ(
                quadPositionConstraintsMeans[2][2] * 1e3,
                quadPositionConstraintsMeans[2][1] * 1e3,
                -quadPositionConstraintsMeans[2][0] * 1e3)

        # Quad tilts constraints means [rad]
        self.quad1TiltConstraintsMean.Set(
                quadTiltConstraintsMeans[0][0],
                quadTiltConstraintsMeans[0][1])
        self.quad2TiltConstraintsMean.Set(
                quadTiltConstraintsMeans[1][0],
                quadTiltConstraintsMeans[1][1])
        self.quad3TiltConstraintsMean.Set(
                quadTiltConstraintsMeans[2][0],
                quadTiltConstraintsMeans[2][1])

        # Quad shifts constraints uncertainties [mm]
        self.quad1PositionConstraintsSigma.SetXYZ(
                quadPositionConstraintsSigmas[0][2] * 1e3 if quadPositionConstraintsSigmas[0][2] != np.inf else -1,
                quadPositionConstraintsSigmas[0][1] * 1e3 if quadPositionConstraintsSigmas[0][1] != np.inf else -1,
                quadPositionConstraintsSigmas[0][0] * 1e3 if quadPositionConstraintsSigmas[0][0] != np.inf else -1)
        self.quad2PositionConstraintsSigma.SetXYZ(
                quadPositionConstraintsSigmas[1][2] * 1e3 if quadPositionConstraintsSigmas[0][2] != np.inf else -1,
                quadPositionConstraintsSigmas[1][1] * 1e3 if quadPositionConstraintsSigmas[0][1] != np.inf else -1,
                quadPositionConstraintsSigmas[1][1] * 1e3 if quadPositionConstraintsSigmas[0][1] != np.inf else -1)
        self.quad3PositionConstraintsSigma.SetXYZ(
                quadPositionConstraintsSigmas[2][2] * 1e3 if quadPositionConstraintsSigmas[0][2] != np.inf else -1,
                quadPositionConstraintsSigmas[2][1] * 1e3 if quadPositionConstraintsSigmas[0][1] != np.inf else -1,
                quadPositionConstraintsSigmas[2][2] * 1e3 if quadPositionConstraintsSigmas[0][2] != np.inf else -1)

        # Quad tilts constraints uncertainties [rad]
        self.quad1TiltConstraintsSigma.Set(
                quadTiltConstraintsSigmas[0][0] if quadTiltConstraintsSigmas[0][0] != np.inf else -1,
                quadTiltConstraintsSigmas[0][1] if quadTiltConstraintsSigmas[0][1] != np.inf else -1)
        self.quad2TiltConstraintsSigma.Set(
                quadTiltConstraintsSigmas[1][0] if quadTiltConstraintsSigmas[0][0] != np.inf else -1,
                quadTiltConstraintsSigmas[1][1] if quadTiltConstraintsSigmas[0][1] != np.inf else -1)
        self.quad3TiltConstraintsSigma.Set(
                quadTiltConstraintsSigmas[2][0] if quadTiltConstraintsSigmas[0][0] != np.inf else -1,
                quadTiltConstraintsSigmas[2][1] if quadTiltConstraintsSigmas[0][1] != np.inf else -1)

        # BPM fit status (true == floating, false == fixed)
        # [bpm0, bpm1, bpm2, bpm3]
        for i in range(4):
            self.bpmFree.push_back(bpmFree[i])

        # Quad shifts fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        for i in range(3):
            self.quadShiftsFree.push_back(quadShiftsFree[i])

        # Quad tilits fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        for i in range(3):
            self.quadTiltsFree.push_back(quadTiltsFree[i])

        # Alignment fit chi2
        self.alignmentLongChi2.value = alignmentLongChi2
        self.alignmentShortChi2.value = alignmentShortChi2

        # Alignment fit ndf
        self.alignmentLongNdf.value = alignmentLongNdf
        self.alignmentShortNdf.value = alignmentShortNdf

        # BPM uncertainties used in the alignment fit
        # in local BPM coordinates
        # [bpm0, bpm1, bpm2, bpm3]
        for k in range(4):
            bpmHitCovData = ROOT.TArrayD(4)
            bpmHitCov = ROOT.TMatrixD(2, 2)
            for i in range(2):
                for j in range(2):
                    bpmHitCovData[i * 2 + j] = bpmCovs[k][i, j] * 1e6
            bpmHitCov.Use(2, 2, bpmHitCovData.GetArray())
            self.bpmHitCovs.push_back(bpmHitCov)

        # Alignment covariance
        dimX, dimY = alignmentCov.shape
        for i in range(dimX):
            for j in range(dimY):
                self.alignmentCov[i][j] = alignmentCov[i, j] * 1e6

        self.tree.Fill()
