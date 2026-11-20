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

from Geometry.GeneralConstraints import GeneralConstraints
from Geometry.FacetConstraints import FacetConstraints
from Beamline.E320Beamline import E320Beamline
from Beamline.E320BeamlineDFD import E320BeamlineDFD
from Beamline.E320BeamlineFDF import E320BeamlineFDF

# --------------------------------------------------------------
# Gradient dependent systematic of the reference orbit.
#
# Per steering condition c the alignment p_c is fitted with the corrected
# model (steering response from the constraints) and the orbit at the
# reference is reconstructed per shot with p_c. The true orbit at the
# reference does not depend on the gradients (the reference is upstream of
# the quads), so the mean orbit of the gradient setting g,
#     a_g = v_c + T w_c(G_g) + noise,
# varies over the settings only through the part of the mismodelling the
# alignment fit is blind to (T w_c, w_c the error of the fitted beam vector
# at BPM0). With the deviations a_g - <a> over the N settings,
#     spread = sum_g (a_g - <a>)(a_g - <a>)^T / (N - 1),
#     noise = sum_g C_g / n_g / N (C_g the shot covariance of setting g),
#     alignment = sum_g (S_g - <S>) V_p (S_g - <S>)^T / N
# (S_g the orbit response to the alignment, V_p the alignment covariance:
# the gradient dependent part of the statistical alignment error, already in
# the alignment covariance), the systematic over the gradient set is
#     C_G = (N - 1) / N (spread - noise) - alignment,
# clipped to positive semi-definite. The part common to all settings is not
# measurable (it is indistinguishable from v_c). The covariances of the
# conditions are averaged.

# --------------------------------------------------------------
def splitSteeringConditions(
        bpm0X: npt.NDArray[np.float64],
        bpm0Y: npt.NDArray[np.float64]) -> list[npt.NDArray[np.int64]]:
    """
    Split the shots into steering conditions, tuned by hand for the scan at hand:
    connected components of the occupied cellSize x cellSize cells
    of the BPM0 (x, y) readings, each split further into runs of
    consecutive shots (gaps above maxGap shots), runs with less than
    minShots shots are dropped

    bpm0X: BPM0 x readings [m]
    bpm0Y: BPM0 y readings [m]

    return: list of the shot indices of the steering conditions
    """
    cellSize = 40 * 1e-6 # [m]
    maxGap = 50 # [shots]
    minShots = 100

    cells = defaultdict(list)
    for idx in range(len(bpm0X)):
        cell = (int(np.floor(bpm0X[idx] / cellSize)), int(np.floor(bpm0Y[idx] / cellSize)))
        cells[cell].append(idx)

    conditions = []
    visited = set()
    for startCell in cells:
        if (startCell in visited):
            continue
        visited.add(startCell)
        stack = [startCell]
        blob = []
        while (len(stack) > 0):
            cell = stack.pop()
            blob.extend(cells[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (cell[0] + dx, cell[1] + dy)
                    if (neighbour in cells and neighbour not in visited):
                        visited.add(neighbour)
                        stack.append(neighbour)

        blob = np.sort(np.array(blob, dtype=np.int64))
        for run in np.split(blob, np.where(np.diff(blob) > maxGap)[0] + 1):
            if (len(run) >= minShots):
                conditions.append(run)

    conditions.sort(key=lambda run: run[0])
    return conditions

# --------------------------------------------------------------
def makeBeamline(plane: str) -> E320Beamline:
    """
    Beamline of one plane configured as in BeamlineAlignmentRunCsv,
    with the reference at the Al foil and the steering response 
    from FacetConstraints (corrected model)

    plane: "Long" (y) or "Short" (x)

    return: beamline
    """
    quadSigmaConstraintShift = FacetConstraints.quadSurveySigmaShift
    quadSigmaConstraintTilt = FacetConstraints.quadSurveySigmaTilt

    beamEnergy = 10 * 1e9 * GeneralConstraints.e

    if (plane == "Long"):
        beamline = E320BeamlineFDF()
        quadTilts = np.array([
            FacetConstraints.quad1Yaw,
            FacetConstraints.quad2Yaw,
            FacetConstraints.quad3Yaw], dtype=np.float64)
        quadShiftConstraintsMean = np.array([
            FacetConstraints.quad1CenterLong,
            FacetConstraints.quad2CenterLong,
            FacetConstraints.quad3CenterLong], dtype=np.float64)
    else:
        beamline = E320BeamlineDFD()
        quadTilts = np.array([
            FacetConstraints.quad1Pitch,
            FacetConstraints.quad2Pitch,
            FacetConstraints.quad3Pitch], dtype=np.float64)
        quadShiftConstraintsMean = np.array([
            FacetConstraints.quad1CenterShort,
            FacetConstraints.quad2CenterShort,
            FacetConstraints.quad3CenterShort], dtype=np.float64)

    quadShiftConstraintsSigma = np.array([
        quadSigmaConstraintShift,
        quadSigmaConstraintShift,
        quadSigmaConstraintShift], dtype=np.float64)
    quadTiltConstraintsSigma = np.array([
        quadSigmaConstraintTilt,
        quadSigmaConstraintTilt,
        quadSigmaConstraintTilt], dtype=np.float64)

    quadLengths = np.array([
        2 * FacetConstraints.quad1HalfPrimary,
        2 * FacetConstraints.quad2HalfPrimary,
        2 * FacetConstraints.quad3HalfPrimary], dtype=np.float64)

    freeToQuadLenght = np.array([
        (FacetConstraints.quad1CenterPrimary - FacetConstraints.quad1HalfPrimary) - FacetConstraints.bpm0CenterPrimary,
        (FacetConstraints.quad2CenterPrimary - FacetConstraints.quad2HalfPrimary) - FacetConstraints.bpm1CenterPrimary,
        (FacetConstraints.quad3CenterPrimary - FacetConstraints.quad3HalfPrimary) - FacetConstraints.bpm2CenterPrimary], dtype=np.float64)
    quadToBPMLenght = np.array([
        FacetConstraints.bpm1CenterPrimary - (FacetConstraints.quad1CenterPrimary + FacetConstraints.quad1HalfPrimary),
        FacetConstraints.bpm2CenterPrimary - (FacetConstraints.quad2CenterPrimary + FacetConstraints.quad2HalfPrimary),
        FacetConstraints.bpm3CenterPrimary - (FacetConstraints.quad3CenterPrimary + FacetConstraints.quad3HalfPrimary)], dtype=np.float64)

    referenceDistance = FacetConstraints.alFoilCenterPrimary - FacetConstraints.bpm0CenterPrimary

    beamline.setBeamEnergy(beamEnergy)
    beamline.setQuadShifts(np.zeros(shape=(3), dtype=np.float64))
    beamline.setQuadShiftsConstraints(quadShiftConstraintsMean, quadShiftConstraintsSigma)
    beamline.setQuadTilts(quadTilts)
    beamline.setQuadTiltsConstraints(quadTilts, quadTiltConstraintsSigma)
    beamline.setQuadLengths(quadLengths)
    beamline.setBpmShifts(np.zeros(shape=(4), dtype=np.float64))
    beamline.setFreeSpaceQuadDistances(freeToQuadLenght)
    beamline.setQuadBPMDistances(quadToBPMLenght)
    beamline.setReferenceDistance(FacetConstraints.bpm0CenterPrimary, referenceDistance)
    if (plane == "Long"):
        beamline.setSteeringResponse(FacetConstraints.steeringResponseLong)
    else:
        beamline.setSteeringResponse(FacetConstraints.steeringResponseShort)
    return beamline

# --------------------------------------------------------------
def estimateConditionSystematic(
        beamline: E320Beamline, 
        shots: npt.NDArray[np.int64], 
        bpmShots: npt.NDArray[np.float64], 
        bpmErrs: npt.NDArray[np.float64], 
        quadGrads: npt.NDArray[np.float64], 
        gradientResolution: float, 
        minShotsPerSetting: int, 
        minSettings: int) -> dict:
    """
    Gradient dependent systematic of the reference orbit for one steering condition

    beamline: beamline of the plane, the alignment parameters are 
        the starting point of the alignment fit
    shots: shot indices of the steering condition
    bpmShots: bpm measurements [m]
    bpmErrs: bpm measurement errors [m]
    quadGrads: quadrupole gradients [T/m]
    gradientResolution: gradient quantisation of the settings [T/m]
    minShotsPerSetting: settings with less shots are dropped
    minSettings: conditions with less settings are dropped

    return: dict with the covariance and its components, empty if the condition is dropped
    """
    iniPars = beamline.getAlignmentPars()
    pars, alignmentCov, chi2, ndf = beamline.optimizeBeamline(bpmShots[shots], bpmErrs[shots], quadGrads[shots])
    if (ndf < 0):
        print("Alignment fit failed, condition dropped")
        beamline.setAlignmentPars(iniPars)
        return {}
    beamline.setAlignmentPars(pars)

    # Reference orbit per shot with the alignment of the condition
    settings = defaultdict(list)
    for idx in shots:
        settings[tuple(np.round(quadGrads[idx] / gradientResolution).astype(int))].append(idx)

    settingGrads = []
    settingShots = []
    means = []
    covs = []
    orbitDers = []
    for settingKey, settingIdx in settings.items():
        if (len(settingIdx) < minShotsPerSetting):
            continue
        refVecs = np.zeros(shape=(len(settingIdx), 2), dtype=np.float64)
        for k, idx in enumerate(settingIdx):
            beamVec = beamline.fitBeamNewtonStep(bpmShots[idx], bpmErrs[idx], quadGrads[idx])[0]
            refVecs[k] = beamline.toReference.ABCD @ beamVec
        grads = np.mean(quadGrads[settingIdx], axis=0)
        settingGrads.append(grads)
        settingShots.append(len(settingIdx))
        means.append(np.mean(refVecs, axis=0))
        covs.append(np.cov(refVecs.T, ddof=1))
        orbitDers.append(beamline.getOrbitResponseMatrix(bpmErrs[settingIdx[0]], grads))
    beamline.setAlignmentPars(iniPars)

    nSettings = len(means)
    if (nSettings < minSettings):
        print(f"{nSettings} settings with at least {minShotsPerSetting} shots, condition dropped")
        return {}

    means = np.array(means)
    settingShots = np.array(settingShots)
    meanOrbit = np.mean(means, axis=0)
    deviations = means - meanOrbit

    # Spread of the setting means over the gradient set
    spread = deviations.T @ deviations / (nSettings - 1)

    # Noise of the setting means (beam jitter and BPM noise)
    noiseCov = np.zeros(shape=(2, 2), dtype=np.float64)
    for g in range(nSettings):
        noiseCov += covs[g] / settingShots[g]
    noiseCov /= nSettings

    # Gradient dependent part of the statistical alignment error, 
    # already in the alignment covariance
    meanOrbitDer = np.mean(np.array(orbitDers), axis=0)
    alignmentTerm = np.zeros(shape=(2, 2), dtype=np.float64)
    for g in range(nSettings):
        derDev = orbitDers[g] - meanOrbitDer
        alignmentTerm += derDev @ alignmentCov @ derDev.T
    alignmentTerm /= nSettings

    # Population covariance over the gradient set, clipped to 
    # positive semi-definite
    cov = (nSettings - 1) / nSettings * (spread - noiseCov) - alignmentTerm
    cov = 0.5 * (cov + cov.T)
    eigVals, eigVecs = np.linalg.eigh(cov)
    covClipped = eigVecs @ np.diag(np.clip(eigVals, 0, None)) @ eigVecs.T

    # Significance of the gradient dependence against the noise of the means
    significanceChi2 = 0.0
    for g in range(nSettings):
        significanceChi2 += deviations[g] @ np.linalg.inv(covs[g] / settingShots[g]) @ deviations[g]
    significanceNdf = 2 * (nSettings - 1)

    print(f"Alignment chi2/ndf {chi2 / ndf:.2f}, {nSettings} settings")
    print("  gradients [T/m]          shots  deviation [um, urad]")
    for g in range(nSettings):
        print(f"  {np.round(settingGrads[g], 3)}  {settingShots[g]:5d}  "
              f"{deviations[g,0] * 1e6:8.2f} {deviations[g,1] * 1e6:8.2f}")
    print(f"  gradient dependence chi2/ndf {significanceChi2:.1f}/{significanceNdf}")
    print(f"  sqrt diag [um, urad]: spread {np.sqrt(np.diagonal(spread)) * 1e6}, "
          f"noise {np.sqrt(np.diagonal(noiseCov)) * 1e6}, alignment {np.sqrt(np.diagonal(alignmentTerm)) * 1e6}")
    print(f"  systematic {np.sqrt(np.diagonal(covClipped)) * 1e6}"
          f"{' (clipped)' if np.any(eigVals < 0) else ''}")

    return {
        "cov": covClipped, 
        "covUnclipped": cov, 
        "spread": spread, 
        "noiseCov": noiseCov, 
        "alignmentTerm": alignmentTerm, 
        "deviations": deviations, 
        "settingGrads": np.array(settingGrads), 
        "settingShots": settingShots, 
        "significanceChi2": significanceChi2, 
        "significanceNdf": significanceNdf, 
        "pars": pars, 
        "alignmentCov": alignmentCov}

# --------------------------------------------------------------
def estimateGradientSystematic(
        beamline: E320Beamline, 
        conditions: list[npt.NDArray[np.int64]], 
        bpmShots: npt.NDArray[np.float64], 
        bpmErrs: npt.NDArray[np.float64], 
        quadGrads: npt.NDArray[np.float64], 
        gradientResolution: float, 
        minShotsPerSetting: int, 
        minSettings: int) -> dict:
    """
    Gradient dependent systematic of the reference orbit of one plane, 
    the mean over the steering conditions of the per condition covariances

    beamline: beamline of the plane (see estimateConditionSystematic)
    conditions: shot indices of the steering conditions
    bpmShots: bpm measurements [m]
    bpmErrs: bpm measurement errors [m]
    quadGrads: quadrupole gradients [T/m]
    gradientResolution: gradient quantisation of the settings [T/m]
    minShotsPerSetting: settings with less shots are dropped
    minSettings: conditions with less settings are dropped

    return: dict with the 2x2 covariance and the per condition results
    """
    results = []
    for c, shots in enumerate(conditions):
        print("-"*50)
        print(f"Condition {c}: {len(shots)} shots")
        result = estimateConditionSystematic(
            beamline, shots, bpmShots, bpmErrs, quadGrads, 
            gradientResolution, minShotsPerSetting, minSettings)
        if (len(result) > 0):
            result["condition"] = c
            results.append(result)

    if (len(results) == 0):
        print("No steering condition with enough gradient settings")
        return {}

    cov = np.mean(np.array([result["cov"] for result in results]), axis=0)

    print("-"*50)
    print(f"{len(results)} of {len(conditions)} conditions used")
    print(f"Gradient systematic at the reference [um, urad]: sqrt diag {np.sqrt(np.diagonal(cov)) * 1e6}, "
          f"correlation {cov[0,1] / np.sqrt(cov[0,0] * cov[1,1]) if cov[0,0] * cov[1,1] > 0 else 0.0:.2f}")

    return {
        "cov": cov, 
        "conditions": results}

# --------------------------------------------------------------
def processRun(
        pathToInFile: str, 
        plane: str, 
        gradientResolution: float, 
        minShotsPerSetting: int, 
        minSettings: int) -> dict:
    inFile = open(pathToInFile)
    dataReader = list(csv.reader(inFile, delimiter=",", quotechar='|'))

    nShots = len(dataReader)

    bpmXs = np.zeros(shape=(nShots, 4), dtype=float)
    bpmYs = np.zeros(shape=(nShots, 4), dtype=float)
    quadGrads = np.zeros(shape=(nShots, 3), dtype=float)
    for idx,row in enumerate(dataReader):
        bpmXs[idx,:] = np.array([row[0], row[1], row[2], row[3]], dtype=float)
        bpmYs[idx,:] = np.array([row[4], row[5], row[6], row[7]], dtype=float)
        quadGrads[idx,:] = np.array([row[8], row[9], row[10]], dtype=float)

    # ------------------------------------------------------
    conditions = splitSteeringConditions(bpmXs[:,0], bpmYs[:,0])
    print(f"{nShots} shots, {len(conditions)} steering conditions")
    for c, shots in enumerate(conditions):
        print(f"  condition {c}: {len(shots)} shots, BPM0 "
              f"({np.mean(bpmXs[shots,0]) * 1e3:.3f}, {np.mean(bpmYs[shots,0]) * 1e3:.3f}) mm")

    # ------------------------------------------------------
    if (plane == "Long"):
        bpmShots = bpmYs
        errs = np.array(
                [FacetConstraints.bpm0ResolutionEstLong, 
                 FacetConstraints.bpm1ResolutionEstLong, 
                 FacetConstraints.bpm2ResolutionEstLong, 
                 FacetConstraints.bpm3ResolutionEstLong])
    else:
        bpmShots = bpmXs
        errs = np.array(
                [FacetConstraints.bpm0ResolutionEstShort, 
                 FacetConstraints.bpm1ResolutionEstShort, 
                 FacetConstraints.bpm2ResolutionEstShort, 
                 FacetConstraints.bpm3ResolutionEstShort])
    bpmErrs = np.tile(errs, (nShots, 1))

    beamline = makeBeamline(plane)
    result = estimateGradientSystematic(
        beamline, conditions, bpmShots, bpmErrs, quadGrads, 
        gradientResolution, minShotsPerSetting, minSettings)
    if (len(result) == 0):
        return result

    # ------------------------------------------------------
    print("-"*50)
    print("Gradient systematic for the constraints file:")
    print(f"    gradientSystematicCov{plane} = [")
    for i in range(2):
        print(f"        [{result['cov'][i,0]:.6e}, {result['cov'][i,1]:.6e}],")
    print("    ]")

    return result

if __name__ == "__main__":
    pathToInFile = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                    "quad_scan_data_facet/sm_quad_scan_steering/outFile.csv")

    processRun(pathToInFile = pathToInFile,
               plane = "Long", 
               gradientResolution = 1e-3, 
               minShotsPerSetting = 10, 
               minSettings = 5)
