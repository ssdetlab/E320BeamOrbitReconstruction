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
from Io.RootOrbitSimFitWriter import RootOrbitSimFitWriter
from Io.E320RootBeamlineAlignmentReader import E320RootBeamlineAlignmentReader
from Types.BeamOrbitSim import BeamOrbitSim

ROOT.gSystem.Load("/home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so")
sys.path.append("../")

# ------------------------------------------------------
pathToDataDir = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_dataInRootFormat/sim/test/alignment_shifts_quad_rots/test/clusters"
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

pathToMagDir = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_dataInRootFormat/sim/test/alignment_shifts_quad_rots/test/magnets"
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
alignmentReaderCfg = E320RootBeamlineAlignmentReader.Config()
alignmentReaderCfg.treeName = "alignment-parameters"
alignmentReaderCfg.filePath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/sim/alignment_shifts_quad_rots/alignment_rots/alignment-parameters.root"

alignmentReader = E320RootBeamlineAlignmentReader(alignmentReaderCfg)
alignmentParsLong = alignmentReader.getAlignmentParsLong()
alignmentParsShort = alignmentReader.getAlignmentParsShort()

# ------------------------------------------------------

alignmentProvider = False
if (alignmentProvider):
    beamEnergy = 10 * 1e9 * GeneralConstraints.e
    
    quadShiftsLong = alignmentParsLong[4:7]
    quadShiftsShort = alignmentParsShort[4:7]
    
    quadLengths = np.array([
        2 * SimConstraints.quad1HalfPrimary,
        2 * SimConstraints.quad2HalfPrimary,
        2 * SimConstraints.quad3HalfPrimary], dtype=np.float64)
    
    bpmShiftsLong = alignmentParsLong[0:4]
    bpmShiftsShort = alignmentParsShort[0:4]
    
    quadTiltsShort = np.array([
        SimConstraints.quad1Pitch,
        SimConstraints.quad2Pitch,
        SimConstraints.quad3Pitch], dtype=np.float64)
    quadTiltsLong = np.array([
        SimConstraints.quad1Yaw,
        SimConstraints.quad2Yaw,
        SimConstraints.quad3Yaw], dtype=np.float64)

    freeToQuadLenght = np.array([
        (SimConstraints.quad1CenterPrimary - SimConstraints.quad1HalfPrimary) - SimConstraints.bpm0CenterPrimary,
        (SimConstraints.quad2CenterPrimary - SimConstraints.quad2HalfPrimary) - SimConstraints.bpm1CenterPrimary,
        (SimConstraints.quad3CenterPrimary - SimConstraints.quad3HalfPrimary) - SimConstraints.bpm2CenterPrimary], dtype=np.float64)
    quadToBPMLenght = np.array([
        SimConstraints.bpm1CenterPrimary - (SimConstraints.quad1CenterPrimary + SimConstraints.quad1HalfPrimary),
        SimConstraints.bpm2CenterPrimary - (SimConstraints.quad2CenterPrimary + SimConstraints.quad2HalfPrimary),
        SimConstraints.bpm3CenterPrimary - (SimConstraints.quad3CenterPrimary + SimConstraints.quad3HalfPrimary)], dtype=np.float64)
    
    referenceDistance = 1515.0 * 1e-3
    # referenceDistance = 0
else:
    beamEnergy = 10 * 1e9 * GeneralConstraints.e
    
    quadShiftsLong = np.array([
        SimConstraints.quad1CenterLong,
        SimConstraints.quad2CenterLong,
        SimConstraints.quad3CenterLong], dtype=np.float64)
    quadShiftsShort = np.array([
        SimConstraints.quad1CenterShort,
        SimConstraints.quad2CenterShort,
        SimConstraints.quad3CenterShort], dtype=np.float64)
    
    quadTiltsShort = np.array([
        SimConstraints.quad1Pitch,
        SimConstraints.quad2Pitch,
        SimConstraints.quad3Pitch], dtype=np.float64)
    quadTiltsLong = np.array([
        SimConstraints.quad1Yaw,
        SimConstraints.quad2Yaw,
        SimConstraints.quad3Yaw], dtype=np.float64)

    quadLengths = np.array([
        2 * SimConstraints.quad1HalfPrimary,
        2 * SimConstraints.quad2HalfPrimary,
        2 * SimConstraints.quad3HalfPrimary], dtype=np.float64)
    
    bpmShiftsLong = np.array([
        SimConstraints.bpm0CenterLong,
        SimConstraints.bpm1CenterLong,
        SimConstraints.bpm2CenterLong,
        SimConstraints.bpm3CenterLong], dtype=np.float64)
    bpmShiftsShort = np.array([
        SimConstraints.bpm0CenterShort,
        SimConstraints.bpm1CenterShort,
        SimConstraints.bpm2CenterShort,
        SimConstraints.bpm3CenterShort], dtype=np.float64)
    
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
print(f"Quad shifts short: {quadShiftsShort}")
print(f"Quad tilts short: {quadTiltsShort}")
print(f"Quad tilts long: {quadTiltsLong}")
print(f"Quad shifts lenghts: {quadLengths}")
print(f"BPM shifts long: {bpmShiftsLong}")
print(f"BPM shifts short: {bpmShiftsShort}")
print(f"Free to quad lengths: {freeToQuadLenght}")
print(f"Quad to BPM lengths: {quadToBPMLenght}")

# ------------------------------------------------------
beamlineX = E320BeamlineDFD()

beamlineX.setBeamEnergy(beamEnergy)
beamlineX.setQuadShifts(quadShiftsShort)
beamlineX.setQuadTilts(quadTiltsShort)
beamlineX.setQuadLengths(quadLengths)
beamlineX.setBpmShifts(bpmShiftsShort)
beamlineX.setFreeSpaceQuadDistances(freeToQuadLenght)
beamlineX.setQuadBPMDistances(quadToBPMLenght)
beamlineX.setReferenceDistance(SimConstraints.bpm0CenterPrimary, referenceDistance)

# ------------------------------------------------------
beamlineY = E320BeamlineFDF()

beamlineY.setBeamEnergy(beamEnergy)
beamlineY.setQuadShifts(quadShiftsLong)
beamlineY.setQuadTilts(quadTiltsLong)
beamlineY.setQuadLengths(quadLengths)
beamlineY.setBpmShifts(bpmShiftsLong)
beamlineY.setFreeSpaceQuadDistances(freeToQuadLenght)
beamlineY.setQuadBPMDistances(quadToBPMLenght)
beamlineY.setReferenceDistance(SimConstraints.bpm0CenterPrimary, referenceDistance)

# ------------------------------------------------------
beamWriterCfg = RootOrbitSimFitWriter.Config()
beamWriterCfg.filePath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/fitted-beam-sim.root"
beamWriterCfg.treeName = "fitted-beam"
beamWriterCfg.bpmPositions = np.array(
        list(
            zip(
                beamlineX.bpmShift, 
                beamlineY.bpmShift, 
                [SimConstraints.bpm0CenterPrimary, 
                 SimConstraints.bpm1CenterPrimary, 
                 SimConstraints.bpm2CenterPrimary, 
                 SimConstraints.bpm3CenterPrimary])))

beamWriter = RootOrbitSimFitWriter(beamWriterCfg)

for eventId in range(eventIdMin, eventIdMax):
    clusters = dataReader.read(eventId)
    magnets = np.array(magReader.read(eventId))
    
    if len(clusters) != 4:
        print(f"Skipping {eventId}: not enough clusters")
        continue

    bpmMeasX = np.array(
            [clusters[0].measurement[0], 
             clusters[1].measurement[0], 
             clusters[2].measurement[0], 
             clusters[3].measurement[0]])
    bpmMeasY = np.array(
            [clusters[0].measurement[1], 
             clusters[1].measurement[1], 
             clusters[2].measurement[1], 
             clusters[3].measurement[1]])

    bpmErrsX = np.array(
            [np.sqrt(clusters[0].cov[0][0]), 
             np.sqrt(clusters[1].cov[0][0]), 
             np.sqrt(clusters[2].cov[0][0]), 
             np.sqrt(clusters[3].cov[0][0])])
    bpmErrsY = np.array(
            [np.sqrt(clusters[0].cov[1][1]), 
             np.sqrt(clusters[1].cov[1][1]), 
             np.sqrt(clusters[2].cov[1][1]), 
             np.sqrt(clusters[3].cov[1][1])])

    fitResX = beamlineX.optimizeBeam(bpmMeasX, bpmErrsX, magnets)
    fitResY = beamlineY.optimizeBeam(bpmMeasY, bpmErrsY, magnets)

    # Get true beam hits
    beamHitsTruth = np.array(
            [clusters[0].trueBeamHit, 
             clusters[1].trueBeamHit, 
             clusters[2].trueBeamHit, 
             clusters[3].trueBeamHit])

    # Get true beam vector at the reference
    originMomentumX = clusters[0].originMomentum[0]
    originMomentumY = clusters[0].originMomentum[1]
    originMomentumZ = clusters[0].originMomentum[2]

    truthThetaX = np.arctan(originMomentumX/originMomentumZ)
    truthThetaY = np.arctan(originMomentumY/originMomentumZ)

    vecTruthX = np.array([clusters[0].trueBeamHit[0], truthThetaX])
    vecTruthY = np.array([clusters[0].trueBeamHit[1], truthThetaY])

    vecTruthX = beamlineX.toReference.ABCD @ vecTruthX
    vecTruthY = beamlineY.toReference.ABCD @ vecTruthY

    referencePositionTruth = np.array([vecTruthX[0], vecTruthY[0], beamlineX.bpm0CenterPrimary + beamlineX.referenceDist])
    referenceMomentumTruth = clusters[0].originMomentum

    orbit = BeamOrbitSim(
            fitResX, 
            fitResY, 
            beamHitsTruth, 
            referencePositionTruth, 
            referenceMomentumTruth, 
            np.linalg.norm(clusters[0].originMomentum), 
            eventId)

    beamWriter.write(orbit, magnets)

    if (eventId % 100 == 0):
        print(f"{eventId}/{eventIdMax - eventIdMin}")
