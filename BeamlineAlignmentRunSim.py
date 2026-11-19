import sys
import numpy as np
import numpy.typing as npt
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

def sampleSurvey(truth: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
    """
    Emulate a survey of the beamline elements by sampling 
    around the true values with the survey precision

    truth: true values of the parameters
    sigma: survey precision

    return: surveyed values
    """
    survey = np.zeros(shape=(len(truth)), dtype=np.float64)
    for i in range(len(truth)):
        survey[i] = truth[i] + sigma * np.random.normal()
    return survey

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
# Survey precision of the emulated survey (placeholders)
surveySigmaShift = 0.1e-3 # [m]
surveySigmaTilt = 1e-3 # [rad]

quadSigmaConstraintShift = surveySigmaShift
quadSigmaConstraintTilt = surveySigmaTilt

# Emulated survey of the quads
quadShiftsSurveyLong = sampleSurvey(
    np.array([
        SimConstraints.quad1CenterLong,
        SimConstraints.quad2CenterLong,
        SimConstraints.quad3CenterLong], dtype=np.float64), 
    surveySigmaShift)
quadShiftsSurveyShort = sampleSurvey(
    np.array([
        SimConstraints.quad1CenterShort,
        SimConstraints.quad2CenterShort,
        SimConstraints.quad3CenterShort], dtype=np.float64), 
    surveySigmaShift)
quadTiltsSurveyLong = sampleSurvey(
    np.array([
        SimConstraints.quad1Yaw,
        SimConstraints.quad2Yaw,
        SimConstraints.quad3Yaw], dtype=np.float64), 
    surveySigmaTilt)
quadTiltsSurveyShort = sampleSurvey(
    np.array([
        SimConstraints.quad1Pitch,
        SimConstraints.quad2Pitch,
        SimConstraints.quad3Pitch], dtype=np.float64), 
    surveySigmaTilt)

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
quadTiltsLong = quadTiltsSurveyLong
quadTiltsShort = quadTiltsSurveyShort

quadLengths = np.array([
    2 * SimConstraints.quad1HalfPrimary,
    2 * SimConstraints.quad2HalfPrimary,
    2 * SimConstraints.quad3HalfPrimary], dtype=np.float64)

quadShiftConstraintsMeanLong = quadShiftsSurveyLong
quadShiftConstraintsSigmaLong = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadShiftConstraintsMeanShort = quadShiftsSurveyShort
quadShiftConstraintsSigmaShort = np.array([
    quadSigmaConstraintShift,
    quadSigmaConstraintShift,
    quadSigmaConstraintShift], dtype=np.float64)

quadTiltConstraintsMeanLong = quadTiltsSurveyLong
quadTiltConstraintsSigmaLong = np.array([
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt,
    quadSigmaConstraintTilt], dtype=np.float64)

quadTiltConstraintsMeanShort = quadTiltsSurveyShort
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
beamlineX.setSteeringResponse(SimConstraints.steeringResponseShort)

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
beamlineY.setSteeringResponse(SimConstraints.steeringResponseLong)

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

# Positions of the beamline elements [x, y, z] [m]
bpmPrimary = np.array([
    SimConstraints.bpm0CenterPrimary,
    SimConstraints.bpm1CenterPrimary,
    SimConstraints.bpm2CenterPrimary,
    SimConstraints.bpm3CenterPrimary], dtype=np.float64)
quadPrimary = np.array([
    SimConstraints.quad1CenterPrimary,
    SimConstraints.quad2CenterPrimary,
    SimConstraints.quad3CenterPrimary], dtype=np.float64)

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
