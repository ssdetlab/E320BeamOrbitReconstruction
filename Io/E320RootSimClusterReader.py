import ROOT
import numpy as np

from Geometry.GeneralConstraints import GeneralConstraints
from Types.SimBPMMeasurement import SimBPMMeasurement

class E320RootSimClusterReader:
    """
    ROOT file reader designed for the ACTS sim data
    """
    class Config:
        """
        Nested configuration struct
        """
        # The names of the input files
        filePaths = []
        # Name of the input tree
        treeName = ""
        # Geometry ID scope
        minGeoId = 0
        maxGeoId = 0

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
            self.dataFile = ROOT.TFile(self.cfg.filePaths[0])
            self.tree = self.dataFile.Get(self.cfg.treeName)
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
        self.tree.GetEntry(0)
        self.eventMap.append([getattr(self.tree, "eventId"), 0, 0])
        for i in range(nEntries):
            self.tree.GetEntry(i)
            eventId = getattr(self.tree, "eventId")
            if eventId != self.eventMap[-1][0]:
                self.eventMap[-1][2] = i
                self.eventMap.append([eventId, i, i])
            if i == nEntries - 1:
                self.eventMap[-1][2] = nEntries
        self.eventMap.sort(key = lambda x: x[0])
        self.tree.SetBranchStatus("*", True)

    def availableEvents(self) -> tuple[int, int]:
        """
        Return the available events range
        """
        return (self.eventMap[0][0], self.eventMap[-1][0])

    def read(self, eventId: int) -> list[SimBPMMeasurement]:
        """
        Read out data from the input stream

        eventId: current event ID
        """
        idxMin, idxMax = 0, 0
        for i in range(len(self.eventMap)):
            if self.eventMap[i][0] == eventId:
                eventId, idxMin, idxMax = self.eventMap[i]
                break

        clusters = []
        for i in range(idxMin, idxMax):
            self.tree.GetEntry(i)
            cluster = SimBPMMeasurement()

            geoId = getattr(self.tree, "geoId")
            if geoId < self.cfg.minGeoId or geoId > self.cfg.maxGeoId:
                continue

            geoCenterLocal = getattr(self.tree, "geoCenterLocal")
            cov = getattr(self.tree, "clusterCov")
            eventId = getattr(self.tree, "eventId")

            cluster.measurement = np.array([geoCenterLocal.Y(), geoCenterLocal.X()]) * 1e-3
            cluster.cov = np.array([[cov(1, 1), cov(1, 0)], [cov(0, 1), cov(0, 0)]]) * 1e-6
            cluster.geoId = geoId
            cluster.eventId = eventId

            truthHits = getattr(self.tree, "trackHitsLocal")
            vertex = getattr(self.tree, "vertex")[0] * 1e-3
            originMomenutm = getattr(self.tree, "originMomentum")[0]
            if len(truthHits) != 1:
                raise RuntimeError("Only one truth hit per BPM is expected")

            truthHits = truthHits[0]
            cluster.trueBeamHit = np.array([truthHits.Y(), truthHits.X()]) * 1e-3
            cluster.originPosition = np.array([-vertex.Z(), vertex.Y(), vertex.X()])
            cluster.originMomentum = np.array([-originMomenutm.Pz(), originMomenutm.Py(), originMomenutm.Px()]) * 1e9 * GeneralConstraints.e

            clusters.append(cluster)

        return clusters
