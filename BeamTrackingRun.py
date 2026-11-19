import sys
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "Io"))
sys.path.insert(0, str(project_dir / "Geometry"))
sys.path.insert(0, str(project_dir / "Beamline"))

import ROOT
from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.FacetConstraints import FacetConstraints
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF
from Io.E320RootDataReader import E320RootDataReader
from Io.RootOrbitFitWriter import RootOrbitFitWriter
from Io.E320RootBeamlineAlignmentReader import E320RootBeamlineAlignmentReader
from Types.BeamOrbit import BeamOrbit

ROOT.gSystem.Load("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_preprocessing/build/libDetEventClusteredDict.so")

def processRun(
        pathToDataDir: str, 
        runPrefix: str,
        dataTreeName: str, 
        dataEventKey: str, 
        pathToAlignmentData: str, 
        alignmentTreeName: str, 
        pathsToAlignmentSamples: list[str], 
        outDirPath: str, 
        outTreeName: str) -> None:
    dataPathList = Path(pathToDataDir).rglob('*.root')
    dataPaths = [str(path) for path in dataPathList]
    dataPaths.sort()
    for path in dataPaths:
        print(path)
    
    dataReaderCfg = E320RootDataReader.Config()
    dataReaderCfg.treeName = dataTreeName
    dataReaderCfg.filePaths = dataPaths
    dataReaderCfg.eventKey = dataEventKey
    
    dataReader = E320RootDataReader(dataReaderCfg)
    
    # ------------------------------------------------------
    alignmentReaderCfg = E320RootBeamlineAlignmentReader.Config()
    alignmentReaderCfg.treeName = alignmentTreeName
    alignmentReaderCfg.filePath = pathToAlignmentData
    
    alignmentReader = E320RootBeamlineAlignmentReader(alignmentReaderCfg)

    # Alignment samples for the orbit shifts at the reference
    alignmentSamplesLong = []
    alignmentSamplesShort = []
    for pathToAlignmentSample in pathsToAlignmentSamples:
        alignmentSampleReaderCfg = E320RootBeamlineAlignmentReader.Config()
        alignmentSampleReaderCfg.treeName = alignmentTreeName
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
        (FacetConstraints.quad3CenterPrimary - FacetConstraints.quad3HalfPrimary) - FacetConstraints.bpm2CenterPrimary], 
                                dtype=np.float64)
    quadToBPMLenght = np.array([
        FacetConstraints.bpm1CenterPrimary - (FacetConstraints.quad1CenterPrimary + FacetConstraints.quad1HalfPrimary),
        FacetConstraints.bpm2CenterPrimary - (FacetConstraints.quad2CenterPrimary + FacetConstraints.quad2HalfPrimary),
        FacetConstraints.bpm3CenterPrimary - (FacetConstraints.quad3CenterPrimary + FacetConstraints.quad3HalfPrimary)], 
                               dtype=np.float64)
    
    referenceDistance = FacetConstraints.alFoilCenterPrimary - FacetConstraints.bpm0CenterPrimary
    
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
    print(f"Reference distance: {referenceDistance}")
    
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
    beamlineY.setBeamBunchCov(
            FacetConstraints.referenceBeamSizeLong, 
            FacetConstraints.referenceBeamDivergenceLong, 
            FacetConstraints.referenceBeamCorrelationLong)
    beamlineY.setAlignmentCov(alignmentReader.getAlignmentCovLong())
    beamlineY.setAlignmentSamples(alignmentSamplesLong)
    beamlineY.setAlignmentSampleRmsThreshold(FacetConstraints.alignmentSampleRmsThreshold)
    
    # ------------------------------------------------------
    outDir = Path(outDirPath + "/" + runPrefix)
    outDir.mkdir(exist_ok = True)
    print(f"Output directory: {outDir}")

    beamWriterCfg = RootOrbitFitWriter.Config()
    beamWriterCfg.filePath = str(outDir) + "/fitted-beam-" + runPrefix + ".root"
    beamWriterCfg.treeName = outTreeName
    beamWriterCfg.bpmPositions = np.array(
            list(
                zip(
                    beamlineX.bpmShift, 
                    beamlineY.bpmShift, 
                    [FacetConstraints.bpm0CenterPrimary, 
                     FacetConstraints.bpm1CenterPrimary, 
                     FacetConstraints.bpm2CenterPrimary, 
                     FacetConstraints.bpm3CenterPrimary])))
    
    beamWriter = RootOrbitFitWriter(beamWriterCfg)
    
    nEntries = dataReader.getNEntries()
    for idx in range(nEntries):
        clusters, magnets = dataReader.read()
        
        if len(clusters) != 4:
            print(f"Skipping {idx}: not enough clusters")
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
                idx,
                clusters[0].trigN,
                clusters[0].runNumber)
    
        beamWriter.write(orbit, magnets)
    
        if (idx % 100 == 0):
            print(f"{idx}/{nEntries}")

if __name__ == "__main__":
    pathToDataDir = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_dataInRootFormat/local_alignment_data/raw/sig"
    
    pathToAlignmentData = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                           "out_data/data/bpm_alignment_doug_survey_quads/est_bpm_res_sm_no_steering/sm_no_steering/"
                           "alignment/alignment-parameters.root")

    # Alignment samples for the orbit shifts at the reference
    # (empty -> no alignment sample contribution)
    pathsToAlignmentSamples = []
    
    outDirPath = "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/out_data/data/global_2026_data"

    runDirsList = Path(pathToDataDir).rglob('data_Run*')
    for path in runDirsList:
        print(path)
        strPath = str(path)
    
        startIdx = strPath.find("data_Run")
        runPrefix = strPath[startIdx:]
        print(runPrefix)

        strPath = strPath + "/ev_data"
        processRun(pathToDataDir = strPath,
                   runPrefix = runPrefix,
                   dataTreeName = "MyTree", 
                   dataEventKey = "event", 
                   pathToAlignmentData = pathToAlignmentData, 
                   alignmentTreeName = "alignment-parameters", 
                   pathsToAlignmentSamples = pathsToAlignmentSamples, 
                   outDirPath = outDirPath, 
                   outTreeName = "fitted-beam")
