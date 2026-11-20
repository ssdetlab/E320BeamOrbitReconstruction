import sys
import csv
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "Io"))
sys.path.insert(0, str(project_dir / "Geometry"))
sys.path.insert(0, str(project_dir / "Beamline"))

from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.FacetConstraints import FacetConstraints
from Geometry.SimConstraints import SimConstraints
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF
from Io.RootOrbitFitWriter import RootOrbitFitWriter
from Io.E320RootBeamlineAlignmentReader import E320RootBeamlineAlignmentReader
from Types.BeamOrbit import BeamOrbit

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
alignmentFilePath = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                     "out_data/data/global_2026_data/beamline_preparations/bpm_res_est/"
                     "sm_no_steering_pre_steering_response/alignment-parameters.root")
alignmentReaderCfg = E320RootBeamlineAlignmentReader.Config()
alignmentReaderCfg.treeName = "alignment-parameters"
alignmentReaderCfg.filePath = alignmentFilePath

alignmentReader = E320RootBeamlineAlignmentReader(alignmentReaderCfg)

# Alignment samples for the orbit shifts at the reference
# (empty -> no alignment sample contribution)
pathsToAlignmentSamples = []

alignmentSamplesLong = []
alignmentSamplesShort = []
for pathToAlignmentSample in pathsToAlignmentSamples:
    alignmentSampleReaderCfg = E320RootBeamlineAlignmentReader.Config()
    alignmentSampleReaderCfg.treeName = alignmentReaderCfg.treeName
    alignmentSampleReaderCfg.filePath = pathToAlignmentSample

    alignmentSampleReader = E320RootBeamlineAlignmentReader(alignmentSampleReaderCfg)
    alignmentSamplesLong.append(alignmentSampleReader.getAlignmentParsLong())
    alignmentSamplesShort.append(alignmentSampleReader.getAlignmentParsShort())
alignmentParsLong = alignmentReader.getAlignmentParsLong()
alignmentParsShort = alignmentReader.getAlignmentParsShort()

# ------------------------------------------------------
beamEnergy = 10 * 1e9 * GeneralConstraints.e

quadShiftsLong = alignmentParsLong[4:7]
quadShiftsShort = alignmentParsShort[4:7]

quadTiltsLong = alignmentParsLong[7:10]
quadTiltsShort = alignmentParsShort[7:10]

quadLengths = np.array([
    2 * FacetConstraints.quad1HalfPrimary,
    2 * FacetConstraints.quad2HalfPrimary,
    2 * FacetConstraints.quad3HalfPrimary], dtype=np.float64)

bpmShiftsLong = alignmentParsLong[0:4]
bpmShiftsShort = alignmentParsShort[0:4]

freeToQuadLenght = np.array([
    (FacetConstraints.quad1CenterPrimary - FacetConstraints.quad1HalfPrimary) - FacetConstraints.bpm0CenterPrimary,
    (FacetConstraints.quad2CenterPrimary - FacetConstraints.quad2HalfPrimary) - FacetConstraints.bpm1CenterPrimary,
    (FacetConstraints.quad3CenterPrimary - FacetConstraints.quad3HalfPrimary) - FacetConstraints.bpm2CenterPrimary], dtype=np.float64)
quadToBPMLenght = np.array([
    FacetConstraints.bpm1CenterPrimary - (FacetConstraints.quad1CenterPrimary + FacetConstraints.quad1HalfPrimary),
    FacetConstraints.bpm2CenterPrimary - (FacetConstraints.quad2CenterPrimary + FacetConstraints.quad2HalfPrimary),
    FacetConstraints.bpm3CenterPrimary - (FacetConstraints.quad3CenterPrimary + FacetConstraints.quad3HalfPrimary)], dtype=np.float64)

# referenceDistance = 1515.0 * 1e-3
referenceDistance = 0

print("-"*50)
print(f"Beam energy: {beamEnergy}")
print(f"Quad shifts long: {quadShiftsLong}")
print(f"Quad shifts short: {quadShiftsShort}")
print(f"Quad tilts long: {quadTiltsLong}")
print(f"Quad tilts short: {quadTiltsShort}")
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
beamlineX.setReferenceDistance(FacetConstraints.bpm0CenterPrimary, referenceDistance)
beamlineX.setSteeringResponse(FacetConstraints.steeringResponseShort)
beamlineX.setGradientSystematicCov(FacetConstraints.gradientSystematicCovShort)
beamlineX.setBeamBunchCov(
        FacetConstraints.referenceBeamSizeShort, 
        FacetConstraints.referenceBeamDivergenceShort, 
        FacetConstraints.referenceBeamCorrelationShort)
beamlineX.setAlignmentCov(alignmentReader.getAlignmentCovShort())
beamlineX.setAlignmentSamples(alignmentSamplesShort)
beamlineX.setAlignmentSampleRmsThreshold(FacetConstraints.alignmentSampleRmsThreshold)

# ------------------------------------------------------
beamlineY = E320BeamlineFDF()

beamlineY.setBeamEnergy(beamEnergy)
beamlineY.setQuadShifts(quadShiftsLong)
beamlineY.setQuadTilts(quadTiltsLong)
beamlineY.setQuadLengths(quadLengths)
beamlineY.setBpmShifts(bpmShiftsLong)
beamlineY.setFreeSpaceQuadDistances(freeToQuadLenght)
beamlineY.setQuadBPMDistances(quadToBPMLenght)
beamlineY.setReferenceDistance(FacetConstraints.bpm0CenterPrimary, referenceDistance)
beamlineY.setSteeringResponse(FacetConstraints.steeringResponseLong)
beamlineY.setGradientSystematicCov(FacetConstraints.gradientSystematicCovLong)
beamlineY.setBeamBunchCov(
        FacetConstraints.referenceBeamSizeLong, 
        FacetConstraints.referenceBeamDivergenceLong, 
        FacetConstraints.referenceBeamCorrelationLong)
beamlineY.setAlignmentCov(alignmentReader.getAlignmentCovLong())
beamlineY.setAlignmentSamples(alignmentSamplesLong)
beamlineY.setAlignmentSampleRmsThreshold(FacetConstraints.alignmentSampleRmsThreshold)

# ------------------------------------------------------
beamWriterCfg = RootOrbitFitWriter.Config()
beamWriterCfg.filePath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/fitted-beam.root"
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

beamWriter = RootOrbitFitWriter(beamWriterCfg)

for eventId,row in enumerate(dataReader):
    bpmMeasX = np.array([row[0], row[1], row[2], row[3]], dtype=float)
    bpmMeasY = np.array([row[4], row[5], row[6], row[7]], dtype=float)
    magnets = np.array([row[8], row[9], row[10]], dtype=float)

    bpmErrsX = np.array(
            [FacetConstraints.bpm0ResolutionEstShort, 
             FacetConstraints.bpm1ResolutionEstShort, 
             FacetConstraints.bpm2ResolutionEstShort, 
             FacetConstraints.bpm3ResolutionEstShort])
    bpmErrsY = np.array(
            [FacetConstraints.bpm0ResolutionEstLong, 
             FacetConstraints.bpm1ResolutionEstLong, 
             FacetConstraints.bpm2ResolutionEstLong, 
             FacetConstraints.bpm3ResolutionEstLong])

    fitResX = beamlineX.optimizeBeam(bpmMeasX, bpmErrsX, magnets)
    fitResY = beamlineY.optimizeBeam(bpmMeasY, bpmErrsY, magnets)

    # Refit of the measurements generated from the estimated orbit 
    # with the BPM resolutions (parametric bootstrap)
    bpmMeasBootstrapX = np.array(
            [fitResX.beamStates[i].bpmPrediction + bpmErrsX[i] * np.random.normal() for i in range(4)])
    bpmMeasBootstrapY = np.array(
            [fitResY.beamStates[i].bpmPrediction + bpmErrsY[i] * np.random.normal() for i in range(4)])

    fitResBootstrapX = beamlineX.optimizeBeam(bpmMeasBootstrapX, bpmErrsX, magnets)
    fitResBootstrapY = beamlineY.optimizeBeam(bpmMeasBootstrapY, bpmErrsY, magnets)

    orbit = BeamOrbit(
            fitResX, 
            fitResY, 
            fitResBootstrapX, 
            fitResBootstrapY, 
            eventId,
            0,
            0)

    magnets = np.array([magnets[0], magnets[1], magnets[2]], dtype=float)
    beamWriter.write(orbit, magnets)

    if (eventId % 100 == 0):
        print(f"{eventId}")

