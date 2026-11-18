import sys
import csv
import numpy as np
import numpy.typing as npt
from pathlib import Path
from collections import defaultdict

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "Io"))
sys.path.insert(0, str(project_dir / "Geometry"))
sys.path.insert(0, str(project_dir / "Beamline"))

import ROOT
from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.SimConstraints import SimConstraints
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF
from Geometry.FacetConstraints import FacetConstraints
from Io.E320RootBeamlineAlignmentReader import E320RootBeamlineAlignmentReader

ROOT.gSystem.Load("/home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so")
sys.path.append("../")

# --------------------------------------------------------------
# BPM resolution estimation from the covariance of the orbit fit residuals.
#
# Per plane, y = m - b0 - B p = A u + eps, Cov(eps) = diag(sigma_j^2).
# The residuals are r = P y = P eps + P B dp, P = I - A K, P A = 0,
# so the beam drops out and the alignment error dp gives an offset
# that is constant within one magnet setting.
# The residuals live in the 2D left null space of A: with the rows
# of L (2x4) an orthonormal basis of it (L A = 0), z = L r = L eps + const and
# Cov(z) = L diag(sigma_j^2) L^T = sum_j sigma_j^2 l_j l_j^T, l_j = L[:, j].
# The 3 independent entries of Cov(z) per magnet setting are linear in the
# 4 unknown sigma_j^2: a linear regression through the origin,
# stacked over the magnet settings (one setting identifies only 3 combinations).

def getResidualSpaceBasis(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Orthonormal basis of the residual space (left null space of A)

    A: 4x2 beam response matrix [m/m, m/rad]

    return: 2x4 matrix L with orthonormal rows, L A = 0
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    return U[:, 2:].T

def getCovRegressors(L: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Regressors of the residual covariance entries on the BPM variances

    L: 2x4 residual space basis

    return: 3x4 matrix X, rows (00, 01, 11) entries of Cov(z), columns BPMs,
            Cov(z)_ab = sum_j X[ab, j] sigma_j^2
    """
    X = np.zeros(shape=(3, 4), dtype=np.float64)
    for j in range(4):
        X[0, j] = L[0, j] * L[0, j]
        X[1, j] = L[0, j] * L[1, j]
        X[2, j] = L[1, j] * L[1, j]
    return X

def getWishartCov(cov: npt.NDArray[np.float64], nEvents: int) -> npt.NDArray[np.float64]:
    """
    Covariance of the (00, 01, 11) entries of a 2x2 sample covariance
    of Gaussian vectors (Wishart), Cov(S_ab, S_cd) = (cov_ac cov_bd + cov_ad cov_bc) / (n - 1)

    cov: 2x2 covariance of the vectors
    nEvents: number of vectors

    return: 3x3 covariance
    """
    idx = [(0, 0), (0, 1), (1, 1)]
    wishartCov = np.zeros(shape=(3, 3), dtype=np.float64)
    for i, (a, b) in enumerate(idx):
        for j, (c, d) in enumerate(idx):
            wishartCov[i, j] = (cov[a, c] * cov[b, d] + cov[a, d] * cov[b, c]) / (nEvents - 1)
    return wishartCov

def fitBpmVariances(
        regressors: list[npt.NDArray[np.float64]],
        sampleCovs: list[npt.NDArray[np.float64]],
        nEvents: list[int],
        nIterations: int = 3) -> dict:
    """
    Generalised least squares fit of the BPM variances to the residual covariances
    of the magnet settings, the Wishart weights are evaluated at the model

    regressors: per setting 3x4 regressor matrix
    sampleCovs: per setting 2x2 sample covariance of z = L r
    nEvents: per setting number of events
    nIterations: number of weight updates

    return: dict with the variances [m^2], their covariance [m^4], chi2, ndf,
            the singular values of the weighted regressors and the per-setting pulls
            of the covariance entries
    """
    nSettings = len(regressors)
    X = np.vstack(regressors)
    s = np.zeros(shape=(3 * nSettings), dtype=np.float64)
    for g in range(nSettings):
        s[3 * g + 0] = sampleCovs[g][0, 0]
        s[3 * g + 1] = sampleCovs[g][0, 1]
        s[3 * g + 2] = sampleCovs[g][1, 1]

    # Start from the unweighted fit, the Wishart weights from the sample covariances
    modelCovs = sampleCovs
    variances = np.array([0])
    weights = np.array([0])
    normalMatrix = np.array([0])
    for _ in range(nIterations + 1):
        weights = np.zeros(shape=(3 * nSettings, 3 * nSettings), dtype=np.float64)
        for g in range(nSettings):
            weights[3 * g:3 * g + 3, 3 * g:3 * g + 3] = np.linalg.inv(getWishartCov(modelCovs[g], nEvents[g]))
        normalMatrix = X.T @ weights @ X
        variances = np.linalg.solve(normalMatrix, X.T @ weights @ s)

        # Model covariances at the current variances,
        # the sample covariance where the model is not positive definite
        modelCovs = []
        for g in range(nSettings):
            fitted = regressors[g] @ variances
            modelCov = np.array([[fitted[0], fitted[1]], [fitted[1], fitted[2]]], dtype=np.float64)
            if (np.any(np.linalg.eigvalsh(modelCov) <= 0)):
                modelCov = sampleCovs[g]
            modelCovs.append(modelCov)

    residuals = s - X @ variances
    chi2 = float(residuals @ weights @ residuals)
    pulls = residuals / np.sqrt(np.linalg.diagonal(np.linalg.inv(weights)))

    # Conditioning: singular values of the whitened regressors
    weightsSqrt = np.linalg.cholesky(weights)
    singularValues = np.linalg.svd(weightsSqrt.T @ X, compute_uv=False)

    return {
        "variances" : variances,
        "cov" : np.linalg.inv(normalMatrix),
        "chi2" : chi2,
        "ndf" : 3 * nSettings - 4,
        "singularValues" : singularValues,
        "pulls" : pulls.reshape(nSettings, 3)}

def processRun(
        pathToInFile: str, 
        pathToAlignmentData: str, 
        alignmentTreeName: str, 
        gradientResolution: float,
        minEventsPerSetting: int) -> dict:
    inFile = open(pathToInFile)
    dataReader = list(csv.reader(inFile, delimiter=",", quotechar='|'))
    
    # ------------------------------------------------------
    alignmentReaderCfg = E320RootBeamlineAlignmentReader.Config()
    alignmentReaderCfg.treeName = alignmentTreeName
    alignmentReaderCfg.filePath = pathToAlignmentData
    
    alignmentReader = E320RootBeamlineAlignmentReader(alignmentReaderCfg)

    alignmentParsLong = alignmentReader.getAlignmentParsLong()
    alignmentParsShort = alignmentReader.getAlignmentParsShort()
    
    # ------------------------------------------------------
    beamEnergy = 10 * 1e9 * GeneralConstraints.e
    
    quadShiftsLong = alignmentParsLong[4:7]
    quadShiftsShort = alignmentParsShort[4:7]

    quadTiltsLong = alignmentParsLong[7:10]
    quadTiltsShort = alignmentParsShort[7:10]
    
    quadLengths = np.array([
        2 * SimConstraints.quad1HalfPrimary,
        2 * SimConstraints.quad2HalfPrimary,
        2 * SimConstraints.quad3HalfPrimary], dtype=np.float64)
    
    bpmShiftsLong = alignmentParsLong[0:4]
    bpmShiftsShort = alignmentParsShort[0:4]
    
    freeToQuadLenght = np.array([
        (SimConstraints.quad1CenterPrimary - SimConstraints.quad1HalfPrimary) - SimConstraints.bpm0CenterPrimary,
        (SimConstraints.quad2CenterPrimary - SimConstraints.quad2HalfPrimary) - SimConstraints.bpm1CenterPrimary,
        (SimConstraints.quad3CenterPrimary - SimConstraints.quad3HalfPrimary) - SimConstraints.bpm2CenterPrimary], 
                                dtype=np.float64)
    quadToBPMLenght = np.array([
        SimConstraints.bpm1CenterPrimary - (SimConstraints.quad1CenterPrimary + SimConstraints.quad1HalfPrimary),
        SimConstraints.bpm2CenterPrimary - (SimConstraints.quad2CenterPrimary + SimConstraints.quad2HalfPrimary),
        SimConstraints.bpm3CenterPrimary - (SimConstraints.quad3CenterPrimary + SimConstraints.quad3HalfPrimary)], 
                               dtype=np.float64)
    
    referenceDistance = SimConstraints.alFoilCenterPrimary - SimConstraints.bpm0CenterPrimary
    
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
    # Fit residuals grouped by the magnet setting,
    # the gradients are quantised with gradientResolution
    residualMapX = defaultdict(list)
    residualMapY = defaultdict(list)
    gradientMap = defaultdict(list)
    for eventId,row in enumerate(dataReader):
        if (float(row[4]) > -1e-3):
            continue
        bpmMeasX = np.array([row[0], row[1], row[2], row[3]], dtype=float)
        bpmMeasY = np.array([row[4], row[5], row[6], row[7]], dtype=float)
        magnets = np.array([row[8], row[9], row[10]], dtype=float)

        bpmErrsX = np.array(
                [FacetConstraints.bpm0ResolutionSurveyShort, 
                 FacetConstraints.bpm1ResolutionSurveyShort, 
                 FacetConstraints.bpm2ResolutionSurceyShort, 
                 FacetConstraints.bpm3ResolutionSurveyShort])
        bpmErrsY = np.array(
                [FacetConstraints.bpm0ResolutionSurveyLong, 
                 FacetConstraints.bpm1ResolutionSurveyLong, 
                 FacetConstraints.bpm2ResolutionSurveyLong, 
                 FacetConstraints.bpm3ResolutionSurveyLong])
    
        fitResX = beamlineX.optimizeBeam(bpmMeasX, bpmErrsX, magnets)
        fitResY = beamlineY.optimizeBeam(bpmMeasY, bpmErrsY, magnets)
    
        settingKey = tuple(np.round(magnets / gradientResolution).astype(int))

        nStates = len(fitResX.beamStates)
        residualsX = np.zeros(shape=(nStates), dtype=np.float64)
        residualsY = np.zeros(shape=(nStates), dtype=np.float64)
        for i in range(nStates):
            stateX = fitResX.beamStates[i]
            stateY = fitResY.beamStates[i]
            residualsX[i] = stateX.bpmMeas - stateX.bpmPrediction
            residualsY[i] = stateY.bpmMeas - stateY.bpmPrediction

        residualMapX[settingKey].append(residualsX)
        residualMapY[settingKey].append(residualsY)
        gradientMap[settingKey].append(magnets)

        if (eventId % 100 == 0):
            print(f"{eventId}")

    # ------------------------------------------------------
    # Regression of the residual covariances on the BPM variances per plane
    results = {}
    for plane, beamline, residualMap in [("Short", beamlineX, residualMapX), ("Long", beamlineY, residualMapY)]:
        regressors = []
        sampleCovs = []
        nEvents = []
        settings = []
        for settingKey, residualList in residualMap.items():
            if (len(residualList) < minEventsPerSetting):
                continue
            grads = np.mean(np.array(gradientMap[settingKey]), axis=0)

            # Residual space at the setting,
            # z = L r removes the beam (L A = 0)
            base = beamline.getResponseBase(grads)
            A = beamline.getBeamResponseMatrix(base, grads)
            L = getResidualSpaceBasis(A)
        
            residuals = np.array(residualList)
            z = residuals @ L.T

            # Sample covariance, the setting mean (alignment offset) removed
            sampleCovs.append(np.cov(z.T))
            regressors.append(getCovRegressors(L))
            nEvents.append(len(residualList))
            settings.append(grads)

        print("-"*50)
        print(f"Plane {plane}: {len(settings)} settings with at least {minEventsPerSetting} events")
        if (len(settings) < 2):
            print(f"Plane {plane}: at least 2 magnet settings are needed, skipping")
            continue

        fit = fitBpmVariances(regressors, sampleCovs, nEvents)
        conditioning = fit["singularValues"][-1] / fit["singularValues"][0]
        if (conditioning < 1e-10):
            print(f"Plane {plane}: the BPM variances are not identifiable from these settings, skipping")
            continue

        variances = fit["variances"]
        varianceErrs = np.sqrt(np.linalg.diagonal(fit["cov"]))
        print(f"Plane {plane}: chi2/ndf = {fit['chi2']:.2f}/{fit['ndf']}, conditioning {conditioning:.2e}")
        for j in range(4):
            if (variances[j] > 0):
                resolution = np.sqrt(variances[j])
                resolutionErr = varianceErrs[j] / (2 * resolution)
                print(f"  BPM{j}: sigma = {resolution * 1e6:.2f} +- {resolutionErr * 1e6:.2f} um")
            else:
                print(f"  BPM{j}: sigma^2 = {variances[j] * 1e12:.2f} +- {varianceErrs[j] * 1e12:.2f} um^2 (not positive)")
        varianceCorr = fit["cov"] / np.outer(varianceErrs, varianceErrs)
        print(f"  sigma^2 correlations:\n{np.array2string(varianceCorr, precision=2)}")
        for g in range(len(settings)):
            print(f"  setting {np.array2string(settings[g], precision=3)}: "
                  f"{nEvents[g]} events, pulls of Cov(z) (00, 01, 11) {np.array2string(fit['pulls'][g], precision=2)}")

        results[plane] = fit

    return results

if __name__ == "__main__":
    pathToInFile = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                    "quad_scan_data_facet/sm_quad_scan_steering/outFile.csv")
    # pathToAlignmentData = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
    #                        "out_data/data/bpm_alignment_doug_survey_quads/survey_bpm_res/"
    #                        "sm_no_steering/alignment/alignment-parameters.root")
    pathToAlignmentData = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                           "out_data/data/test/"
                           "alignment-parameters.root")

    processRun(pathToInFile = pathToInFile,
               pathToAlignmentData = pathToAlignmentData, 
               alignmentTreeName = "alignment-parameters", 
               gradientResolution = 1e-3,
               minEventsPerSetting = 10)

