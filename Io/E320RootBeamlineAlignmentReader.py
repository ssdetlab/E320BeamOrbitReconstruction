import ROOT

import numpy as np
import numpy.typing as npt

class E320RootBeamlineAlignmentReader:
    """
    Alignment parameters reader

    Reads the alignment written by E320RootBeamlineAlignmentWriter
    and converts it back to the beamline frame and units.
    The convention is [x, y, z] for the position vectors [m],
    [pitch, yaw] for quad angles [rad]. Unconstrained parameters
    (stored as -1) are returned as np.inf.
    """
    class Config:
        """
        Nested configuration struct
        """
        # Name of the input tree
        treeName = ""
        # Name of the input file
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

        self.file = ROOT.TFile(self.cfg.filePath)
        self.tree = self.file.Get(self.cfg.treeName)

        self.tree.GetEntry(0)

        def getSigma(value: float, scale: float) -> float:
            """
            Convert a stored constraint uncertainty

            value: stored uncertainty, -1 -> unconstrained
            scale: unit conversion factor

            return: uncertainty, np.inf -> unconstrained
            """
            if (value == -1):
                return np.inf
            return value * scale

        def getPosition(branchName: str) -> npt.NDArray[np.float64]:
            """
            Read a position vector stored as [z, y, -x] [mm]

            branchName: name of the TVector3 branch

            return: position [x, y, z] [m]
            """
            vec = getattr(self.tree, branchName)
            position = np.zeros(shape=(3), dtype=float)
            position[0] = -vec.Z() * 1e-3 # x
            position[1] = vec.Y() * 1e-3  # y
            position[2] = vec.X() * 1e-3  # z
            return position

        def getPositionSigma(branchName: str) -> npt.NDArray[np.float64]:
            """
            Read a position uncertainty vector stored as [z, y, x] [mm]

            branchName: name of the TVector3 branch

            return: position uncertainty [x, y, z] [m]
            """
            vec = getattr(self.tree, branchName)
            sigma = np.zeros(shape=(3), dtype=float)
            sigma[0] = getSigma(vec.Z(), 1e-3) # x
            sigma[1] = getSigma(vec.Y(), 1e-3) # y
            sigma[2] = getSigma(vec.X(), 1e-3) # z
            return sigma

        def getAngles(branchName: str) -> npt.NDArray[np.float64]:
            """
            Read a quad angles vector stored as [pitch, yaw] [rad]

            branchName: name of the TVector2 branch

            return: angles [pitch, yaw] [rad]
            """
            vec = getattr(self.tree, branchName)
            angles = np.zeros(shape=(2), dtype=float)
            angles[0] = vec.X() # pitch
            angles[1] = vec.Y() # yaw
            return angles

        def getAnglesSigma(branchName: str) -> npt.NDArray[np.float64]:
            """
            Read a quad angles uncertainty vector stored as [pitch, yaw] [rad]

            branchName: name of the TVector2 branch

            return: angles uncertainty [pitch, yaw] [rad]
            """
            vec = getattr(self.tree, branchName)
            sigma = np.zeros(shape=(2), dtype=float)
            sigma[0] = getSigma(vec.X(), 1.0) # pitch
            sigma[1] = getSigma(vec.Y(), 1.0) # yaw
            return sigma

        # -------------------------------------------------------------
        # Members

        # BPM positions [m]
        self.bpmPositions = [
            getPosition("bpm0Position"),
            getPosition("bpm1Position"),
            getPosition("bpm2Position"),
            getPosition("bpm3Position")]

        # BPM constraints means [m]
        self.bpmPositionConstraintsMeans = [
            getPosition("bpm0PositionConstraintsMean"),
            getPosition("bpm1PositionConstraintsMean"),
            getPosition("bpm2PositionConstraintsMean"),
            getPosition("bpm3PositionConstraintsMean")]

        # BPM constraints uncertainties [m]
        self.bpmPositionConstraintsSigmas = [
            getPositionSigma("bpm0PositionConstraintsSigma"),
            getPositionSigma("bpm1PositionConstraintsSigma"),
            getPositionSigma("bpm2PositionConstraintsSigma"),
            getPositionSigma("bpm3PositionConstraintsSigma")]

        # Quad positions [m]
        self.quadPositions = [
            getPosition("quad1Position"),
            getPosition("quad2Position"),
            getPosition("quad3Position")]

        # Quad tilts [rad]
        self.quadTilts = [
            getAngles("quad1Tilt"),
            getAngles("quad2Tilt"),
            getAngles("quad3Tilt")]

        # Quad shifts constraints means [m]
        self.quadPositionConstraintsMeans = [
            getPosition("quad1PositionConstraintsMean"),
            getPosition("quad2PositionConstraintsMean"),
            getPosition("quad3PositionConstraintsMean")]

        # Quad tilts constraints means [rad]
        self.quadTiltConstraintsMeans = [
            getAngles("quad1TiltConstraintsMean"),
            getAngles("quad2TiltConstraintsMean"),
            getAngles("quad3TiltConstraintsMean")]

        # Quad shifts constraints uncertainties [m]
        self.quadPositionConstraintsSigmas = [
            getPositionSigma("quad1PositionConstraintsSigma"),
            getPositionSigma("quad2PositionConstraintsSigma"),
            getPositionSigma("quad3PositionConstraintsSigma")]

        # Quad tilts constraints uncertainties [rad]
        self.quadTiltConstraintsSigmas = [
            getAnglesSigma("quad1TiltConstraintsSigma"),
            getAnglesSigma("quad2TiltConstraintsSigma"),
            getAnglesSigma("quad3TiltConstraintsSigma")]

        # BPM fit status (true == floating, false == fixed)
        # [bpm0, bpm1, bpm2, bpm3]
        bpmFree = getattr(self.tree, "bpmFree")
        self.bpmFree = np.zeros(shape=(4), dtype=bool)
        for i in range(4):
            self.bpmFree[i] = bpmFree[i]

        # Quad shifts fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        quadShiftsFree = getattr(self.tree, "quadShiftsFree")
        self.quadShiftsFree = np.zeros(shape=(3), dtype=bool)
        for i in range(3):
            self.quadShiftsFree[i] = quadShiftsFree[i]

        # Quad tilts fit status (true == floating, false == fixed)
        # [quad1, quad2, quad3]
        quadTiltsFree = getattr(self.tree, "quadTiltsFree")
        self.quadTiltsFree = np.zeros(shape=(3), dtype=bool)
        for i in range(3):
            self.quadTiltsFree[i] = quadTiltsFree[i]

        # Alignment fit chi2
        self.alignmentLongChi2 = getattr(self.tree, "alignmentLongChi2")
        self.alignmentShortChi2 = getattr(self.tree, "alignmentShortChi2")

        # Alignment fit ndf
        self.alignmentLongNdf = getattr(self.tree, "alignmentLongNdf")
        self.alignmentShortNdf = getattr(self.tree, "alignmentShortNdf")

        # Alignment fit covariance [m, rad]
        # (stored in [mm, rad], per direction: 7 shifts, then 3 tilts)
        nParsDirection = 10
        nShiftsDirection = 7
        alignmentCov = getattr(self.tree, "alignmentCov")
        nRows = alignmentCov.GetNrows()
        nCols = alignmentCov.GetNcols()
        covScale = np.zeros(shape=(nRows), dtype=float)
        for i in range(nRows):
            if (i % nParsDirection < nShiftsDirection):
                covScale[i] = 1e-3 # shift
            else:
                covScale[i] = 1.0  # tilt
        self.alignmentCov = np.zeros(shape=(nRows, nCols), dtype=float)
        for i in range(nRows):
            for j in range(nCols):
                self.alignmentCov[i, j] = alignmentCov[i][j] * covScale[i] * covScale[j]

        # BPM uncertainties used in the alignment fit
        # in local BPM coordinates [m^2]
        # [bpm0, bpm1, bpm2, bpm3]
        bpmHitCovs = getattr(self.tree, "bpmHitCovs")
        self.bpmCovs = []
        for k in range(4):
            bpmCov = np.zeros(shape=(2, 2), dtype=float)
            for i in range(2):
                for j in range(2):
                    bpmCov[i, j] = bpmHitCovs[k][i][j] * 1e-6
            self.bpmCovs.append(bpmCov)

    def getBeamlineAlignment(self):
        """
        Get the alignment in the order of the
        E320RootBeamlineAlignmentWriter.write arguments

        return: BPM positions [m],
            BPM constraints means [m],
            BPM constraints uncertainties [m],
            quad positions [m],
            quad tilts [rad],
            quad shifts constraints means [m],
            quad tilts constraints means [rad],
            quad shifts constraints uncertainties [m],
            quad tilts constraints uncertainties [rad],
            BPM fit status,
            quad shifts fit status,
            quad tilts fit status,
            alignment fit chi2 (long),
            alignment fit chi2 (short),
            alignment fit ndf (long),
            alignment fit ndf (short),
            alignment fit covariance in the format of
                E320RootBeamlineAlignmentWriter.write [m, rad],
            BPM uncertainties used in the alignment fit [m^2]
        """
        return (self.bpmPositions,
                self.bpmPositionConstraintsMeans,
                self.bpmPositionConstraintsSigmas,
                self.quadPositions,
                self.quadTilts,
                self.quadPositionConstraintsMeans,
                self.quadTiltConstraintsMeans,
                self.quadPositionConstraintsSigmas,
                self.quadTiltConstraintsSigmas,
                self.bpmFree,
                self.quadShiftsFree,
                self.quadTiltsFree,
                self.alignmentLongChi2,
                self.alignmentShortChi2,
                self.alignmentLongNdf,
                self.alignmentShortNdf,
                self.alignmentCov,
                self.bpmCovs)
