import ROOT
import numpy as np

class E320RootBeamlineAlignmentReader:
    class Config:
        # Name of the output tree
        treeName = ""
        # Name of the output file
        filePath = ""

    # @brief constructor
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        if not self.cfg.filePath:
            raise RuntimeError("Missing file name")
        if not self.cfg.treeName:
            raise RuntimeError("Missing tree name")

        self.file = ROOT.TFile(self.cfg.filePath)
        self.tree = self.file.Get(self.cfg.treeName)

        self.tree.GetEntry(0)

        # BPM positions [mm]
        self.bpmCenterPrimary = np.array(
                [getattr(self.tree, "bpm0CenterPrimary"),
                 getattr(self.tree, "bpm1CenterPrimary"),
                 getattr(self.tree, "bpm2CenterPrimary"),
                 getattr(self.tree, "bpm3CenterPrimary")])

        self.bpmCenterLong = np.array(
                [getattr(self.tree, "bpm0CenterLong"),
                 getattr(self.tree, "bpm1CenterLong"),
                 getattr(self.tree, "bpm2CenterLong"),
                 getattr(self.tree, "bpm3CenterLong")])

        self.bpmCenterShort = np.array(
                [getattr(self.tree, "bpm0CenterShort"),
                 getattr(self.tree, "bpm1CenterShort"),
                 getattr(self.tree, "bpm2CenterShort"),
                 getattr(self.tree, "bpm3CenterShort")])

        # Quad positions [mm]
        self.quadCenterPrimary = np.array(
                [getattr(self.tree, "quad1CenterPrimary"),
                 getattr(self.tree, "quad2CenterPrimary"),
                 getattr(self.tree, "quad3CenterPrimary")])
         
        self.quadCenterLong = np.array(
                [getattr(self.tree, "quad1CenterLong"),
                 getattr(self.tree, "quad2CenterLong"),
                 getattr(self.tree, "quad3CenterLong")])
         
        self.quadCenterShort = np.array(
                [getattr(self.tree, "quad1CenterShort"),
                 getattr(self.tree, "quad2CenterShort"),
                 getattr(self.tree, "quad3CenterShort")])
         
    def getBeamlineAlignment(self):
        return (self.bpmCenterPrimary * 1e-3,
                self.bpmCenterLong * 1e-3,
                self.bpmCenterShort * 1e-3,
                self.quadCenterPrimary * 1e-3,
                self.quadCenterLong * 1e-3,
                self.quadCenterShort * 1e-3)
