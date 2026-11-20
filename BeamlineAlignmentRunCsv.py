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
import csv

# ------------------------------------------------------
inFilePath = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
              "quad_scan_data_facet/sm_quad_scan_no_steering/outFile.csv")
inFile = open(inFilePath)
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

# ------------------------------------------------------
quadSigmaConstraintShift = FacetConstraints.quadSurveySigmaShift
quadSigmaConstraintTilt = FacetConstraints.quadSurveySigmaTilt

beamEnergy = 10 * 1e9 * GeneralConstraints.e

quadShiftsLong = np.array([
    0,
    0,
    0], dtype=np.float64)
quadShiftsShort = np.array([
    0,
    0,
    0], dtype=np.float64)

# Quad tilts, fixed in the alignment fit at the survey values [rad]
quadTiltsLong = np.array([
    FacetConstraints.quad1Yaw,
    FacetConstraints.quad2Yaw,
    FacetConstraints.quad3Yaw], dtype=np.float64)
quadTiltsShort = np.array([
    FacetConstraints.quad1Pitch,
    FacetConstraints.quad2Pitch,
    FacetConstraints.quad3Pitch], dtype=np.float64)

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
beamlineX.setSteeringResponse(FacetConstraints.steeringResponseShort)

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
beamlineY.setSteeringResponse(FacetConstraints.steeringResponseLong)

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
            [FacetConstraints.bpm0ResolutionEstShort, 
             FacetConstraints.bpm1ResolutionEstShort, 
             FacetConstraints.bpm2ResolutionEstShort, 
             FacetConstraints.bpm3ResolutionEstShort])
    bpmShotsX.append(measX)
    bpmErrsX.append(errsX)

    measY = np.array([row[4], row[5], row[6], row[7]], dtype=float)
    errsY = np.array(
            [FacetConstraints.bpm0ResolutionEstLong, 
             FacetConstraints.bpm1ResolutionEstLong, 
             FacetConstraints.bpm2ResolutionEstLong, 
             FacetConstraints.bpm3ResolutionEstLong])
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

# Positions of the beamline elements [x, y, z] [m]
bpmPrimary = np.array([
    FacetConstraints.bpm0CenterPrimary,
    FacetConstraints.bpm1CenterPrimary,
    FacetConstraints.bpm2CenterPrimary,
    FacetConstraints.bpm3CenterPrimary], dtype=np.float64)
quadPrimary = np.array([
    FacetConstraints.quad1CenterPrimary,
    FacetConstraints.quad2CenterPrimary,
    FacetConstraints.quad3CenterPrimary], dtype=np.float64)

bpmPositions = []
bpmPositionConstraintsMeans = []
bpmPositionConstraintsSigmas = []
for i in range(4):
    bpmPositions.append(np.array([alignmentParsX[i], alignmentParsY[i], bpmPrimary[i]]))
    bpmPositionConstraintsMeans.append(np.array([
        beamlineX.bpmConstraintMean[i], 
        beamlineY.bpmConstraintMean[i], 
        bpmPrimary[i]]))
    bpmPositionConstraintsSigmas.append(np.array([
        beamlineX.bpmConstraintSigma[i], 
        beamlineY.bpmConstraintSigma[i], 
        np.inf]))

quadPositions = []
quadTilts = []
quadPositionConstraintsMeans = []
quadTiltConstraintsMeans = []
quadPositionConstraintsSigmas = []
quadTiltConstraintsSigmas = []
for i in range(3):
    quadPositions.append(np.array([alignmentParsX[4 + i], alignmentParsY[4 + i], quadPrimary[i]]))
    quadTilts.append(np.array([alignmentParsX[7 + i], alignmentParsY[7 + i]]))
    quadPositionConstraintsMeans.append(np.array([
        beamlineX.quadShiftsConstraintMean[i], 
        beamlineY.quadShiftsConstraintMean[i], 
        quadPrimary[i]]))
    quadTiltConstraintsMeans.append(np.array([
        beamlineX.quadTiltsConstraintMean[i], 
        beamlineY.quadTiltsConstraintMean[i]]))
    quadPositionConstraintsSigmas.append(np.array([
        beamlineX.quadShiftsConstraintSigma[i], 
        beamlineY.quadShiftsConstraintSigma[i], 
        np.inf]))
    quadTiltConstraintsSigmas.append(np.array([
        beamlineX.quadTiltsConstraintSigma[i], 
        beamlineY.quadTiltsConstraintSigma[i]]))

# Alignment fit covariance [Short, Long] [m, rad]
alignmentCov = np.zeros(shape=(20, 20), dtype=np.float64)
alignmentCov[0:10, 0:10] = alignmentCovX
alignmentCov[10:20, 10:20] = alignmentCovY

# BPM uncertainties used in the alignment fit (mean over the shots) [m^2]
bpmCovs = []
for i in range(4):
    bpmCovs.append(np.array([
        [np.mean(bpmErrsX[:,i]**2), 0.0], 
        [0.0, np.mean(bpmErrsY[:,i]**2)]]))

alignmentWriter.write(
        bpmPositions, 
        bpmPositionConstraintsMeans, 
        bpmPositionConstraintsSigmas, 
        quadPositions, 
        quadTilts, 
        quadPositionConstraintsMeans, 
        quadTiltConstraintsMeans, 
        quadPositionConstraintsSigmas, 
        quadTiltConstraintsSigmas, 
        beamlineX.bpmFree, 
        beamlineX.quadShiftsFree, 
        beamlineX.quadTiltsFree, 
        chi2Y, 
        chi2X, 
        ndfY, 
        ndfX, 
        alignmentCov, 
        bpmCovs)
