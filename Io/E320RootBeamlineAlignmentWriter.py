import ROOT

import numpy as np
import numpy.typing as npt
from ctypes import c_int, c_double

class E320RootBeamlineAlignmentWriter:
    """
    Alignment parameters writer
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
            [bpm0, bpm1, bpm2, bpm3, quad1, quad2, quad3, quad1Tilt, quad2Tilt, quad3Tilt]_Short X 
            [bpm0, bpm1, bpm2, bpm3, quad1, quad2, quad3, quad1Tilt, quad2Tilt, quad3Tilt]_Long, [m, rad]
        bpmCovs: BPM covariances used in the alignment fit in the format: 
            [bpm0_x X bpm0_y, bpm1_x X bpm1_y, bpm2_x X bpm2_y, bpm3_x X bpm3_y], [m]
        """
        def getSigma(value: float, scale: float) -> float:
            """
            Convert a constraint uncertainty for writing

            value: uncertainty, np.inf -> unconstrained
            scale: unit conversion factor

            return: stored uncertainty, -1 -> unconstrained
            """
            if (value == np.inf):
                return -1
            return value * scale

        def setPosition(vec: ROOT.TVector3, position: npt.NDArray[np.float64]) -> None:
            """
            Set a position vector, stored as [z, y, -x] [mm]

            vec: TVector3 member to be set
            position: position [x, y, z] [m]
            """
            vec.SetXYZ(
                    position[2] * 1e3,
                    position[1] * 1e3,
                    -position[0] * 1e3)

        def setPositionSigma(vec: ROOT.TVector3, sigma: npt.NDArray[np.float64]) -> None:
            """
            Set a position uncertainty vector, stored as [z, y, x] [mm]

            vec: TVector3 member to be set
            sigma: position uncertainty [x, y, z] [m], np.inf -> unconstrained
            """
            vec.SetXYZ(
                    getSigma(sigma[2], 1e3),
                    getSigma(sigma[1], 1e3),
                    getSigma(sigma[0], 1e3))

        def setAngles(vec: ROOT.TVector2, angles: npt.NDArray[np.float64]) -> None:
            """
            Set a quad angles vector, stored as [pitch, yaw] [rad]

            vec: TVector2 member to be set
            angles: angles [pitch, yaw] [rad]
            """
            vec.Set(
                    angles[0],
                    angles[1])

        def setAnglesSigma(vec: ROOT.TVector2, sigma: npt.NDArray[np.float64]) -> None:
            """
            Set a quad angles uncertainty vector, stored as [pitch, yaw] [rad]

            vec: TVector2 member to be set
            sigma: angles uncertainty [pitch, yaw] [rad], np.inf -> unconstrained
            """
            vec.Set(
                    getSigma(sigma[0], 1.0),
                    getSigma(sigma[1], 1.0))

        # Clear the vectors
        self.bpmHitCovs.clear()
        self.bpmFree.clear()
        self.quadShiftsFree.clear()
        self.quadTiltsFree.clear()

        # BPM positions [mm]
        setPosition(self.bpm0Position, bpmPositions[0])
        setPosition(self.bpm1Position, bpmPositions[1])
        setPosition(self.bpm2Position, bpmPositions[2])
        setPosition(self.bpm3Position, bpmPositions[3])

        # BPM constraints means [mm]
        setPosition(self.bpm0PositionConstraintsMean, bpmPositionConstraintsMeans[0])
        setPosition(self.bpm1PositionConstraintsMean, bpmPositionConstraintsMeans[1])
        setPosition(self.bpm2PositionConstraintsMean, bpmPositionConstraintsMeans[2])
        setPosition(self.bpm3PositionConstraintsMean, bpmPositionConstraintsMeans[3])

        # BPM constraints uncertainties [mm]
        setPositionSigma(self.bpm0PositionConstraintsSigma, bpmPositionConstraintsSigmas[0])
        setPositionSigma(self.bpm1PositionConstraintsSigma, bpmPositionConstraintsSigmas[1])
        setPositionSigma(self.bpm2PositionConstraintsSigma, bpmPositionConstraintsSigmas[2])
        setPositionSigma(self.bpm3PositionConstraintsSigma, bpmPositionConstraintsSigmas[3])

        # Quad positions [mm]
        setPosition(self.quad1Position, quadPositions[0])
        setPosition(self.quad2Position, quadPositions[1])
        setPosition(self.quad3Position, quadPositions[2])

        # Quad tilts [rad]
        setAngles(self.quad1Tilt, quadTilts[0])
        setAngles(self.quad2Tilt, quadTilts[1])
        setAngles(self.quad3Tilt, quadTilts[2])

        # Quad shifts constraints means [mm]
        setPosition(self.quad1PositionConstraintsMean, quadPositionConstraintsMeans[0])
        setPosition(self.quad2PositionConstraintsMean, quadPositionConstraintsMeans[1])
        setPosition(self.quad3PositionConstraintsMean, quadPositionConstraintsMeans[2])

        # Quad tilts constraints means [rad]
        setAngles(self.quad1TiltConstraintsMean, quadTiltConstraintsMeans[0])
        setAngles(self.quad2TiltConstraintsMean, quadTiltConstraintsMeans[1])
        setAngles(self.quad3TiltConstraintsMean, quadTiltConstraintsMeans[2])

        # Quad shifts constraints uncertainties [mm]
        setPositionSigma(self.quad1PositionConstraintsSigma, quadPositionConstraintsSigmas[0])
        setPositionSigma(self.quad2PositionConstraintsSigma, quadPositionConstraintsSigmas[1])
        setPositionSigma(self.quad3PositionConstraintsSigma, quadPositionConstraintsSigmas[2])

        # Quad tilts constraints uncertainties [rad]
        setAnglesSigma(self.quad1TiltConstraintsSigma, quadTiltConstraintsSigmas[0])
        setAnglesSigma(self.quad2TiltConstraintsSigma, quadTiltConstraintsSigmas[1])
        setAnglesSigma(self.quad3TiltConstraintsSigma, quadTiltConstraintsSigmas[2])

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

        # Alignment covariance [mm, rad]
        # (per direction: 7 shifts [m] -> [mm], then 3 tilts [rad])
        nParsDirection = 10
        nShiftsDirection = 7
        dimX, dimY = alignmentCov.shape
        covScale = np.zeros(shape=(dimX), dtype=float)
        for i in range(dimX):
            if (i % nParsDirection < nShiftsDirection):
                covScale[i] = 1e3 # shift
            else:
                covScale[i] = 1.0 # tilt
        for i in range(dimX):
            for j in range(dimY):
                self.alignmentCov[i][j] = alignmentCov[i, j] * covScale[i] * covScale[j]

        self.tree.Fill()
