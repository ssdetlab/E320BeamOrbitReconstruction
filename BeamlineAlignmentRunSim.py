import sys
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "Io"))
sys.path.insert(0, str(project_dir / "Geometry"))
sys.path.insert(0, str(project_dir / "Beamline"))

import ROOT
from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.SimConstraints import SimConstraints
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF
from Io.E320RootMagneticFieldReader import E320RootMagneticFieldReader
from Io.E320RootSimClusterReader import E320RootSimClusterReader
from Io.E320RootBeamlineAlignmentWriter import E320RootBeamlineAlignmentWriter

ROOT.gSystem.Load("/home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so")
sys.path.append("../")

# ------------------------------------------------------
pathToDataDir = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_dataInRootFormat/sim/test/alignment_shifts_quad_rots/alignment/clusters"
dataPathList = Path(pathToDataDir).rglob('*.root')
dataPaths = [str(path) for path in dataPathList]
for path in dataPaths:
    print(path)

dataReaderCfg = E320RootSimClusterReader.Config()
dataReaderCfg.treeName = "clusters"
dataReaderCfg.minGeoId = 30
dataReaderCfg.maxGeoId = 33
dataReaderCfg.filePaths = dataPaths

dataReader = E320RootSimClusterReader(dataReaderCfg)
print(dataReader.availableEvents())

pathToMagDir = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_dataInRootFormat/sim/test/alignment_shifts_quad_rots/alignment/magnets"
magPathList = Path(pathToMagDir).rglob('*.root')
magPaths = [str(path) for path in magPathList]
for path in magPaths:
    print(path)

magReaderCfg = E320RootMagneticFieldReader.Config()
magReaderCfg.treeName = "magnets"
magReaderCfg.filePaths = magPaths

magReader = E320RootMagneticFieldReader(magReaderCfg)
print(magReader.availableEvents())

eventIdMin, eventIdMax = dataReader.availableEvents()
eventIdMax = eventIdMax + 1

# ------------------------------------------------------
quadSigmaConstraintShift = 0.1e-3
quadSigmaConstraintTilt = 1e-3

beamEnergy = 10 * 1e9 * GeneralConstraints.e

quadShiftsLong = np.array([
    0,
    0,
    0], dtype=np.float64)
quadShiftsShort = np.array([
    0,
    0,
    0], dtype=np.float64)

quadTiltsLong = np.array([
    0,
    0,
    0], dtype=np.float64)
quadTiltsShort = np.array([
    0,
    0,
    0], dtype=np.float64)

quadLengths = np.array([
    2 * SimConstraints.quad1HalfPrimary,
    2 * SimConstraints.quad2HalfPrimary,
    2 * SimConstraints.quad3HalfPrimary], dtype=np.float64)

quadShiftConstraintsMeanLong = np.array([
    SimConstraints.quad1CenterLong + quadSigmaConstraintShift * np.random.normal(),
    SimConstraints.quad2CenterLong + quadSigmaConstraintShift * np.random.normal(),
    SimConstraints.quad3CenterLong + quadSigmaConstraintShift * np.random.normal()], dtype=np.float64)
quadShiftConstraintsSigmaLong = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadShiftConstraintsMeanShort = np.array([
    SimConstraints.quad1CenterShort + quadSigmaConstraintShift * np.random.normal(),
    SimConstraints.quad2CenterShort + quadSigmaConstraintShift * np.random.normal(),
    SimConstraints.quad3CenterShort + quadSigmaConstraintShift * np.random.normal()], dtype=np.float64)
quadShiftConstraintsSigmaShort = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadTiltConstraintsMeanLong = np.array([
    SimConstraints.quad1Yaw + quadSigmaConstraintTilt * np.random.normal(),
    SimConstraints.quad2Yaw + quadSigmaConstraintTilt * np.random.normal(),
    SimConstraints.quad3Yaw + quadSigmaConstraintTilt * np.random.normal()], dtype=np.float64)
quadTiltConstraintsSigmaLong = np.array([
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt], dtype=np.float64)

quadTiltConstraintsMeanShort = np.array([
    SimConstraints.quad1Pitch + quadSigmaConstraintTilt * np.random.normal(),
    SimConstraints.quad2Pitch + quadSigmaConstraintTilt * np.random.normal(),
    SimConstraints.quad3Pitch + quadSigmaConstraintTilt * np.random.normal()], dtype=np.float64)
quadTiltConstraintsSigmaShort = np.array([
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt], dtype=np.float64)

bpmShiftsLong = np.array([
    0,
    0,
    0,
    0], dtype=np.float64)
bpmShiftsShort = np.array([
    0,
    0,
    0,
    0], dtype=np.float64)

freeToQuadLenght = np.array([
    (SimConstraints.quad1CenterPrimary - SimConstraints.quad1HalfPrimary) - SimConstraints.bpm0CenterPrimary,
    (SimConstraints.quad2CenterPrimary - SimConstraints.quad2HalfPrimary) - SimConstraints.bpm1CenterPrimary,
    (SimConstraints.quad3CenterPrimary - SimConstraints.quad3HalfPrimary) - SimConstraints.bpm2CenterPrimary], dtype=np.float64)
quadToBPMLenght = np.array([
    SimConstraints.bpm1CenterPrimary - (SimConstraints.quad1CenterPrimary + SimConstraints.quad1HalfPrimary),
    SimConstraints.bpm2CenterPrimary - (SimConstraints.quad2CenterPrimary + SimConstraints.quad2HalfPrimary),
    SimConstraints.bpm3CenterPrimary - (SimConstraints.quad3CenterPrimary + SimConstraints.quad3HalfPrimary)], dtype=np.float64)

referenceDistance = 1515.0 * 1e-3
# referenceDistance = -5 * 1e-3

print("-"*50)
print(f"Beam energy: {beamEnergy}")
print(f"Quad shifts long: {quadShiftsLong}")
print(f"Quad shifts constraints long: {quadShiftConstraintsMeanLong} +/- {quadShiftConstraintsSigmaLong}")
print(f"Quad shifts short: {quadShiftsShort}")
print(f"Quad shifts constraints short: {quadShiftConstraintsMeanShort} +/- {quadShiftConstraintsSigmaShort}")
print(f"Quad tilts long: {quadTiltsLong}")
print(f"Quad tilts constraints long: {quadTiltConstraintsMeanLong} +/- {quadTiltConstraintsSigmaLong}")
print(f"Quad tilts short: {quadTiltsShort}")
print(f"Quad tilts constraints short: {quadTiltConstraintsMeanShort} +/- {quadTiltConstraintsSigmaShort}")
print(f"Quad lenghts: {quadLengths}")
print(f"BPM shifts long: {bpmShiftsLong}")
print(f"BPM shifts short: {bpmShiftsShort}")
print(f"Free to quad lengths: {freeToQuadLenght}")
print(f"Quad to BPM lengths: {quadToBPMLenght}")

# ------------------------------------------------------
beamlineX = E320BeamlineDFD()

beamlineX.setBeamEnergy(beamEnergy)
beamlineX.setQuadShifts(quadShiftsShort)
beamlineX.setQuadShiftsConstraints(quadShiftConstraintsMeanShort, quadShiftConstraintsSigmaShort)
beamlineX.setQuadTilts(quadTiltsShort)
beamlineX.setQuadTiltsConstraints(quadTiltConstraintsMeanShort, quadTiltConstraintsSigmaShort)
beamlineX.setQuadLengths(quadLengths)
beamlineX.setBpmShifts(bpmShiftsShort)
beamlineX.setFreeSpaceQuadDistances(freeToQuadLenght)
beamlineX.setQuadBPMDistances(quadToBPMLenght)
beamlineX.setReferenceDistance(SimConstraints.bpm0CenterPrimary, referenceDistance)

# ------------------------------------------------------
beamlineY = E320BeamlineFDF()

beamlineY.setBeamEnergy(beamEnergy)
beamlineY.setQuadShifts(quadShiftsLong)
beamlineY.setQuadShiftsConstraints(quadShiftConstraintsMeanLong, quadShiftConstraintsSigmaLong)
beamlineY.setQuadTilts(quadTiltsLong)
beamlineY.setQuadTiltsConstraints(quadTiltConstraintsMeanLong, quadTiltConstraintsSigmaLong)
beamlineY.setQuadLengths(quadLengths)
beamlineY.setBpmShifts(bpmShiftsLong)
beamlineY.setFreeSpaceQuadDistances(freeToQuadLenght)
beamlineY.setQuadBPMDistances(quadToBPMLenght)
beamlineY.setReferenceDistance(SimConstraints.bpm0CenterPrimary, referenceDistance)

bpmShotsX = []
bpmErrsX = []

bpmShotsY = []
bpmErrsY = []

quadGrads = []
for eventId in range(eventIdMin, eventIdMax):
    clusters = dataReader.read(eventId)
    magnets = np.array(magReader.read(eventId))
    quadGrads.append(magnets)

    measX = np.array(
            [clusters[0].measurement[0], 
             clusters[1].measurement[0], 
             clusters[2].measurement[0], 
             clusters[3].measurement[0]])
    errsX = np.array(
            [np.sqrt(clusters[0].cov[0][0]), 
             np.sqrt(clusters[1].cov[0][0]), 
             np.sqrt(clusters[2].cov[0][0]), 
             np.sqrt(clusters[3].cov[0][0])])
    bpmShotsX.append(measX)
    bpmErrsX.append(errsX)

    measY = np.array(
            [clusters[0].measurement[1], 
             clusters[1].measurement[1], 
             clusters[2].measurement[1], 
             clusters[3].measurement[1]])
    errsY = np.array(
            [np.sqrt(clusters[0].cov[1][1]), 
             np.sqrt(clusters[1].cov[1][1]), 
             np.sqrt(clusters[2].cov[1][1]), 
             np.sqrt(clusters[3].cov[1][1])])
    bpmShotsY.append(measY)
    bpmErrsY.append(errsY)

quadGrads = np.array(quadGrads, dtype=np.float64)

bpmShotsX = np.array(bpmShotsX, dtype=np.float64)
bpmErrsX = np.array(bpmErrsX, dtype=np.float64)
alignmentParsX, alignmentCovX, chi2X, ndfX = beamlineX.optimizeBeamline(bpmShotsX, bpmErrsX, quadGrads)

bpmShotsY = np.array(bpmShotsY, dtype=np.float64)
bpmErrsY = np.array(bpmErrsY, dtype=np.float64)
alignmentParsY, alignmentCovY, chi2Y, ndfY = beamlineY.optimizeBeamline(bpmShotsY, bpmErrsY, quadGrads)

# ------------------------------------------------------
alignmentWriterCfg = E320RootBeamlineAlignmentWriter.Config()
alignmentWriterCfg.treeName = "alignment-parameters"
alignmentWriterCfg.filePath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/alignment-parameters.root"

alignmentWriter = E320RootBeamlineAlignmentWriter(alignmentWriterCfg)
alignmentWriter.write(
        np.array([
            SimConstraints.bpm0CenterPrimary,
            SimConstraints.bpm1CenterPrimary,
            SimConstraints.bpm2CenterPrimary,
            SimConstraints.bpm3CenterPrimary]),
        np.array([
            alignmentParsY[0],
            alignmentParsY[1],
            alignmentParsY[2],
            alignmentParsY[3]]),
        np.array([
            alignmentParsX[0],
            alignmentParsX[1],
            alignmentParsX[2],
            alignmentParsX[3]]),
        np.array([
            SimConstraints.quad1CenterPrimary,
            SimConstraints.quad2CenterPrimary,
            SimConstraints.quad3CenterPrimary]),
        np.array([
            alignmentParsY[4],
            alignmentParsY[5],
            alignmentParsY[6]]),
        np.array([
            alignmentParsX[4],
            alignmentParsX[5],
            alignmentParsX[6]]))
