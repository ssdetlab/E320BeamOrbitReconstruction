import sys
import numpy as np
from pathlib import Path

from Io.E320RootBeamlineAlignmentWriter import E320RootBeamlineAlignmentWriter

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "Io"))
sys.path.insert(0, str(project_dir / "Geometry"))
sys.path.insert(0, str(project_dir / "Beamline"))

from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.FacetConstraints import FacetConstraints
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF
from Io.E320RootBeamlineAlignmentReader import E320RootBeamlineAlignmentReader
import csv

# ------------------------------------------------------
inFile = open("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/daniel_sample_tanya/outFile.csv")
dataReader = list(csv.reader(inFile, delimiter=",", quotechar='|'))

nShots = len(dataReader)

bpmXs = np.zeros(shape=(nShots, 4), dtype=float)
bpmYs = np.zeros(shape=(nShots, 4), dtype=float)
quadGrads = np.zeros(shape=(nShots, 3), dtype=float)
for idx,row in enumerate(dataReader):
    bpmXs[idx,:] = np.array([row[0], row[1], row[2], row[3]], dtype=float)
    bpmYs[idx,:] = np.array([row[4], row[5], row[6], row[7]], dtype=float)
    quadGrads[idx,:] = np.array([row[8], row[9], row[10]], dtype=float)

print(bpmXs)
print(quadGrads)

# # ------------------------------------------------------
# alignmentReaderCfg = E320RootBeamlineAlignmentReader.Config()
# alignmentReaderCfg.treeName = "alignment-parameters"
# alignmentReaderCfg.filePath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/data/quad_bpm_alignment/daniel_sample/alignment/alignment-parameters.root"
# 
# alignmentReader = E320RootBeamlineAlignmentReader(alignmentReaderCfg)
# bpmCenterPrimary, bpmCenterLong, bpmCenterShort, quadCenterPrimary, quadCenterLong, quadCenterShort = alignmentReader.getBeamlineAlignment()

# ------------------------------------------------------
# quadSigmaConstraintShift = 0.1 * 1e-3
# quadSigmaConstraintTilt = 1e-3

quadSigmaConstraintShift = np.inf
quadSigmaConstraintTilt = np.inf

beamEnergy = 10 * 1e9 * GeneralConstraints.e

quadShiftsLong = np.array([
    0,
    0,
    0], dtype=np.float64)
quadShiftsShort = np.array([
    0,
    0,
    0], dtype=np.float64)

# quadShiftsLong = np.array([
#     FacetConstraints.quad1CenterLong,
#     FacetConstraints.quad2CenterLong,
#     FacetConstraints.quad3CenterLong], dtype=np.float64)
# quadShiftsShort = np.array([
#     FacetConstraints.quad1CenterShort,
#     FacetConstraints.quad2CenterShort,
#     FacetConstraints.quad3CenterShort], dtype=np.float64)

# quadShiftsLong = quadCenterLong
# quadShiftsShort = quadCenterShort

quadTiltsLong = np.array([
    0,
    0,
    0], dtype=np.float64)
quadTiltsShort = np.array([
    0,
    0,
    0], dtype=np.float64)

quadLengths = np.array([
    2 * FacetConstraints.quad1HalfPrimary,
    2 * FacetConstraints.quad2HalfPrimary,
    2 * FacetConstraints.quad3HalfPrimary], dtype=np.float64)

quadShiftConstraintsMeanLong = np.array([
    FacetConstraints.quad1CenterLong,
    FacetConstraints.quad2CenterLong,
    FacetConstraints.quad3CenterLong], dtype=np.float64)
quadShiftConstraintsSigmaLong = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadShiftConstraintsMeanShort = np.array([
    FacetConstraints.quad1CenterShort,
    FacetConstraints.quad2CenterShort,
    FacetConstraints.quad3CenterShort], dtype=np.float64)
quadShiftConstraintsSigmaShort = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadTiltConstraintsMeanLong = np.array([
    FacetConstraints.quad1Yaw,
    FacetConstraints.quad2Yaw,
    FacetConstraints.quad3Yaw], dtype=np.float64)
quadTiltConstraintsSigmaLong = np.array([
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt], dtype=np.float64)

quadTiltConstraintsMeanShort = np.array([
    FacetConstraints.quad1Pitch,
    FacetConstraints.quad2Pitch,
    FacetConstraints.quad3Pitch], dtype=np.float64)
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

# bpmShiftsLong = bpmCenterLong
# bpmShiftsShort = bpmCenterShort

freeToQuadLenght = np.array([
    (FacetConstraints.quad1CenterPrimary - FacetConstraints.quad1HalfPrimary) - FacetConstraints.bpm0CenterPrimary,
    (FacetConstraints.quad2CenterPrimary - FacetConstraints.quad2HalfPrimary) - FacetConstraints.bpm1CenterPrimary,
    (FacetConstraints.quad3CenterPrimary - FacetConstraints.quad3HalfPrimary) - FacetConstraints.bpm2CenterPrimary], dtype=np.float64)
quadToBPMLenght = np.array([
    FacetConstraints.bpm1CenterPrimary - (FacetConstraints.quad1CenterPrimary + FacetConstraints.quad1HalfPrimary),
    FacetConstraints.bpm2CenterPrimary - (FacetConstraints.quad2CenterPrimary + FacetConstraints.quad2HalfPrimary),
    FacetConstraints.bpm3CenterPrimary - (FacetConstraints.quad3CenterPrimary + FacetConstraints.quad3HalfPrimary)], dtype=np.float64)

referenceDistance = 0

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
beamlineX.setReferenceDistance(FacetConstraints.bpm0CenterPrimary, referenceDistance)

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
beamlineY.setReferenceDistance(FacetConstraints.bpm0CenterPrimary, referenceDistance)

# ------------------------------------------------------
bpmShotsX = []
bpmErrsX = []

bpmShotsY = []
bpmErrsY = []

quadGrads = []

for eventId,row in enumerate(dataReader):
    magnets = np.array([row[8], row[9], row[10]], dtype=float)
    quadGrads.append(magnets)

    measX = np.array([row[0], row[1], row[2], row[3]], dtype=float)
    errsX = np.array(
            [FacetConstraints.bpm0ResolutionShort, 
             FacetConstraints.bpm1ResolutionShort, 
             FacetConstraints.bpm2ResolutionShort, 
             FacetConstraints.bpm3ResolutionShort])
    bpmShotsX.append(measX)
    bpmErrsX.append(errsX)

    measY = np.array([row[4], row[5], row[6], row[7]], dtype=float)
    errsY = np.array(
            [FacetConstraints.bpm0ResolutionLong, 
             FacetConstraints.bpm1ResolutionLong, 
             FacetConstraints.bpm2ResolutionLong, 
             FacetConstraints.bpm3ResolutionLong])
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
            FacetConstraints.bpm0CenterPrimary,
            FacetConstraints.bpm1CenterPrimary,
            FacetConstraints.bpm2CenterPrimary,
            FacetConstraints.bpm3CenterPrimary]),
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
            FacetConstraints.quad1CenterPrimary,
            FacetConstraints.quad2CenterPrimary,
            FacetConstraints.quad3CenterPrimary]),
        np.array([
            alignmentParsY[4],
            alignmentParsY[5],
            alignmentParsY[6]]),
        np.array([
            alignmentParsX[4],
            alignmentParsX[5],
            alignmentParsX[6]]))
