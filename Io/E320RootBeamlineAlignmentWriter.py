import ROOT

from ctypes import c_double

class E320RootBeamlineAlignmentWriter:
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

        self.file = ROOT.TFile(self.cfg.filePath, "RECREATE");
        self.tree = ROOT.TTree(self.cfg.treeName, self.cfg.treeName);

        # -------------------------------------------------------------
        # Members

        # BPM positions [mm]
        self.bpm0CenterPrimary = c_double(0)
        self.bpm1CenterPrimary = c_double(0)
        self.bpm2CenterPrimary = c_double(0)
        self.bpm3CenterPrimary = c_double(0)

        self.bpm0CenterLong = c_double(0)
        self.bpm1CenterLong = c_double(0)
        self.bpm2CenterLong = c_double(0)
        self.bpm3CenterLong = c_double(0)

        self.bpm0CenterShort = c_double(0)
        self.bpm1CenterShort = c_double(0)
        self.bpm2CenterShort = c_double(0)
        self.bpm3CenterShort = c_double(0)

        # Quad positions [mm]
        self.quad1CenterPrimary = c_double(0)
        self.quad2CenterPrimary = c_double(0)
        self.quad3CenterPrimary = c_double(0)

        self.quad1CenterLong = c_double(0)
        self.quad2CenterLong = c_double(0)
        self.quad3CenterLong = c_double(0)

        self.quad1CenterShort = c_double(0)
        self.quad2CenterShort = c_double(0)
        self.quad3CenterShort = c_double(0)

        # -------------------------------------------------------------
        # Tree branches

        # BPM positions [mm]
        self.tree.Branch("bpm0CenterPrimary", self.bpm0CenterPrimary, "bpm0CenterPrimary/D")
        self.tree.Branch("bpm1CenterPrimary", self.bpm1CenterPrimary, "bpm1CenterPrimary/D")
        self.tree.Branch("bpm2CenterPrimary", self.bpm2CenterPrimary, "bpm2CenterPrimary/D")
        self.tree.Branch("bpm3CenterPrimary", self.bpm3CenterPrimary, "bpm3CenterPrimary/D")

        self.tree.Branch("bpm0CenterLong", self.bpm0CenterLong, "bpm0CenterLong/D")
        self.tree.Branch("bpm1CenterLong", self.bpm1CenterLong, "bpm1CenterLong/D")
        self.tree.Branch("bpm2CenterLong", self.bpm2CenterLong, "bpm2CenterLong/D")
        self.tree.Branch("bpm3CenterLong", self.bpm3CenterLong, "bpm3CenterLong/D")

        self.tree.Branch("bpm0CenterShort", self.bpm0CenterShort, "bpm0CenterShort/D")
        self.tree.Branch("bpm1CenterShort", self.bpm1CenterShort, "bpm1CenterShort/D")
        self.tree.Branch("bpm2CenterShort", self.bpm2CenterShort, "bpm2CenterShort/D")
        self.tree.Branch("bpm3CenterShort", self.bpm3CenterShort, "bpm3CenterShort/D")

        # Quad positions [mm]
        self.tree.Branch("quad1CenterPrimary", self.quad1CenterPrimary, "quad1CenterPrimary/D")
        self.tree.Branch("quad2CenterPrimary", self.quad2CenterPrimary, "quad2CenterPrimary/D")
        self.tree.Branch("quad3CenterPrimary", self.quad3CenterPrimary, "quad3CenterPrimary/D")

        self.tree.Branch("quad1CenterLong", self.quad1CenterLong, "quad1CenterLong/D")
        self.tree.Branch("quad2CenterLong", self.quad2CenterLong, "quad2CenterLong/D")
        self.tree.Branch("quad3CenterLong", self.quad3CenterLong, "quad3CenterLong/D")

        self.tree.Branch("quad1CenterShort", self.quad1CenterShort, "quad1CenterShort/D")
        self.tree.Branch("quad2CenterShort", self.quad2CenterShort, "quad2CenterShort/D")
        self.tree.Branch("quad3CenterShort", self.quad3CenterShort, "quad3CenterShort/D")

    def __del__(self):
        self.file.Write()
        self.file.Close()

    def write(
            self, 
            bpmCenterPrimary, 
            bpmCenterLong, 
            bpmCenterShort, 
            quadCenterPrimary, 
            quadCenterLong, 
            quadCenterShort):
        
        # BPM positions [mm]
        self.bpm0CenterPrimary.value = bpmCenterPrimary[0] * 1e3
        self.bpm1CenterPrimary.value = bpmCenterPrimary[1] * 1e3
        self.bpm2CenterPrimary.value = bpmCenterPrimary[2] * 1e3
        self.bpm3CenterPrimary.value = bpmCenterPrimary[3] * 1e3

        self.bpm0CenterLong.value = bpmCenterLong[0] * 1e3
        self.bpm1CenterLong.value = bpmCenterLong[1] * 1e3
        self.bpm2CenterLong.value = bpmCenterLong[2] * 1e3
        self.bpm3CenterLong.value = bpmCenterLong[3] * 1e3

        self.bpm0CenterShort.value = bpmCenterShort[0] * 1e3
        self.bpm1CenterShort.value = bpmCenterShort[1] * 1e3
        self.bpm2CenterShort.value = bpmCenterShort[2] * 1e3
        self.bpm3CenterShort.value = bpmCenterShort[3] * 1e3

        # Quad positions [mm]
        self.quad1CenterPrimary.value = quadCenterPrimary[0] * 1e3
        self.quad2CenterPrimary.value = quadCenterPrimary[1] * 1e3
        self.quad3CenterPrimary.value = quadCenterPrimary[2] * 1e3

        self.quad1CenterLong.value = quadCenterLong[0] * 1e3
        self.quad2CenterLong.value = quadCenterLong[1] * 1e3
        self.quad3CenterLong.value = quadCenterLong[2] * 1e3

        self.quad1CenterShort.value = quadCenterShort[0] * 1e3
        self.quad2CenterShort.value = quadCenterShort[1] * 1e3
        self.quad3CenterShort.value = quadCenterShort[2] * 1e3

        self.tree.Fill()
