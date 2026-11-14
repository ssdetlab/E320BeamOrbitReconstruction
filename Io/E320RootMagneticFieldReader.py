import ROOT
import numpy as np

class E320RootMagneticFieldReader:
    """
    ROOT file reader designed for the ACTS sim magnetic field data
    """
    class Config:
        """
        Nested configuration struct
        """
        # The names of the input files
        filePaths = []
        # Name of the input tree
        treeName = ""

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

        self.tree.SetBranchStatus("*", False)
        nEntries = self.tree.GetEntries()
        self.tree.SetBranchStatus("eventId", True)

        # Go through all entries and store the position of the events
        for i in range(nEntries):
            self.tree.GetEntry(i)
            eventId = getattr(self.tree, "eventId")
            self.eventMap.append([eventId, i])
        self.eventMap.sort(key = lambda x: x[0])
        self.tree.SetBranchStatus("*", True)

    def availableEvents(self) -> tuple[int, int]:
        """
        Return the available events range
        """
        return (self.eventMap[0][0], self.eventMap[-1][0])

    def read(self, eventId: int):
        """
        Read out data from the input stream

        eventId: current event ID
        """
        idx = 0
        for i in range(len(self.eventMap)):
            if self.eventMap[i][0] == eventId:
                eventId, idx = self.eventMap[i]
                break

        magFields = []
        self.tree.GetEntry(idx)

        quad1Grad = np.abs(getattr(self.tree, "quad1Grad"))
        quad2Grad = np.abs(getattr(self.tree, "quad2Grad"))
        quad3Grad = np.abs(getattr(self.tree, "quad3Grad"))

        magFields = [quad1Grad, quad2Grad, quad3Grad]

        return magFields

