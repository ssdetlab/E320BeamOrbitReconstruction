import ROOT
import numpy as np

from Types.BPMMeasurement import BPMMeasurement
from Geometry.FacetConstraints import FacetConstraints

class E320RootDataReader:
    """
    ROOT file reader designed for the EUDAQ2 format
    """
    class Config:
        """
        Nested configuration struct
        """
        # The names of the input files
        filePaths = []
        # Name of the input tree
        treeName = ""
        # The keys we have in the ROOT file
        eventKey = ""

    def __init__(self, cfg: Config) -> None:
        """
        Constructor

        cfg: configuration
        """
        self.cfg = cfg
        self.eventMap = []
  
        if not self.cfg.filePaths:
            raise RuntimeError("Missing file name")
        if not self.cfg.treeName:
            raise RuntimeError("Missing tree name")

        if len(self.cfg.filePaths) == 1:
            self.file = ROOT.TFile(self.cfg.filePaths[0])
            self.tree = self.file.Get(self.cfg.treeName)
        else:
            self.chainOwner = ROOT.TChain(self.cfg.treeName)
            # Add the files to the chain
            for path in self.cfg.filePaths:
                self.chainOwner.Add(path)
            self.tree = self.chainOwner

        self.nEntries = self.tree.GetEntries()
        self.currentIdx = 0

        self.bpm_3156_x_key = "BPMS:LI20:3156:X"
        self.bpm_3156_y_key = "BPMS:LI20:3156:Y"

        self.bpm_quad0_3218_x_key = "BPMS:LI20:3218:X"
        self.bpm_quad0_3218_y_key = "BPMS:LI20:3218:Y"

        self.bpm_quad1_3265_x_key = "BPMS:LI20:3265:X"
        self.bpm_quad1_3265_y_key = "BPMS:LI20:3265:Y"

        self.bpm_quad2_3315_x_key = "BPMS:LI20:3315:X"
        self.bpm_quad2_3315_y_key = "BPMS:LI20:3315:Y"

        self.li20_lgps_3141_bact = "LI20:LGPS:3141:BACT"
        self.li20_lgps_3261_bact = "LI20:LGPS:3261:BACT"
        self.li20_lgps_3091_bact = "LI20:LGPS:3091:BACT"

    def getNEntries(self) -> int:
          return self.nEntries

    def getPv(self, detEvent, key: str):
          return float(str(detEvent.ev_epics_frame.pv_map.at(key).value))

    def convertEpicsQuad(self, val: float) -> float:
      return val * 0.1;

    def read(self) -> tuple[list[BPMMeasurement], list[float]]:
        """
        Read out data from the input stream
        """
        
        if (self.currentIdx >= self.nEntries):
            raise RuntimeError("Current idx is greater than the number of entries")

        clusters = []

        self.tree.GetEntry(self.currentIdx)
        detEvent = getattr(self.tree, self.cfg.eventKey)

        bpm3156Ev = BPMMeasurement()
        bpm3156Ev.measurement = np.array(
                [self.getPv(detEvent, self.bpm_3156_x_key), 
                 self.getPv(detEvent, self.bpm_3156_y_key)]) * 1e-3
        bpm3156Ev.cov = np.array([
                [FacetConstraints.bpm0ResolutionEstShort**2, 0.0], 
                [0.0, FacetConstraints.bpm0ResolutionEstLong**2]])
        bpm3156Ev.geoId = 3156
        bpm3156Ev.trigN = detEvent.trg_n
        bpm3156Ev.runNumber = detEvent.run_number
        clusters.append(bpm3156Ev)

        bpm3218Ev = BPMMeasurement()
        bpm3218Ev.measurement = np.array(
                [self.getPv(detEvent, self.bpm_quad0_3218_x_key), 
                 self.getPv(detEvent, self.bpm_quad0_3218_y_key)]) * 1e-3
        bpm3218Ev.cov = np.array([
                [FacetConstraints.bpm1ResolutionEstShort**2, 0.0], 
                [0.0, FacetConstraints.bpm1ResolutionEstLong**2]])
        bpm3218Ev.geoId = 3218
        bpm3218Ev.trigN = detEvent.trg_n
        bpm3218Ev.runNumber = detEvent.run_number
        clusters.append(bpm3218Ev)

        bpm3265Ev = BPMMeasurement()
        bpm3265Ev.measurement = np.array(
                [self.getPv(detEvent, self.bpm_quad1_3265_x_key), 
                 self.getPv(detEvent, self.bpm_quad1_3265_y_key)]) * 1e-3
        bpm3265Ev.cov = np.array([
                [FacetConstraints.bpm2ResolutionSurceyShort**2, 0.0], 
                [0.0, FacetConstraints.bpm2ResolutionEstLong**2]])
        bpm3265Ev.geoId = 3265
        bpm3265Ev.trigN = detEvent.trg_n
        bpm3265Ev.runNumber = detEvent.run_number
        clusters.append(bpm3265Ev)

        bpm3315Ev = BPMMeasurement()
        bpm3315Ev.measurement = np.array(
                [self.getPv(detEvent, self.bpm_quad2_3315_x_key), 
                 self.getPv(detEvent, self.bpm_quad2_3315_y_key)]) * 1e-3
        bpm3315Ev.cov = np.array([
                [FacetConstraints.bpm3ResolutionEstShort**2, 0.0], 
                [0.0, FacetConstraints.bpm3ResolutionEstLong**2]])
        bpm3315Ev.geoId = 3315
        bpm3315Ev.trigN = detEvent.trg_n
        bpm3315Ev.runNumber = detEvent.run_number
        clusters.append(bpm3315Ev)

        quad1Grad = np.abs(self.convertEpicsQuad(self.getPv(detEvent, self.li20_lgps_3141_bact)))
        quad2Grad = np.abs(self.convertEpicsQuad(self.getPv(detEvent, self.li20_lgps_3261_bact)))
        quad3Grad = np.abs(self.convertEpicsQuad(self.getPv(detEvent, self.li20_lgps_3091_bact)))
        magnets = [quad1Grad, quad2Grad, quad3Grad]
        
        self.currentIdx += 1

        return (clusters, magnets)
