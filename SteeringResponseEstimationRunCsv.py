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
# Steering response estimation from a steering scan.
#
# The scan is split into steering conditions c (BPM0 blobs). Per condition
# the alignment is fitted with the uncorrected model (steering response 0),
# giving p_c with the covariance Sigma_c, and the beam vector at BPM0 is
# fitted per shot with p_c, giving the mean position y_c of the condition.
# The alignment that absorbs a mismodelling linear in the beam position is
#     p_c = p0 + j (y_c - <y>)
# with j the position column of the steering response J (the angle column
# stays 0: one steering direction identifies one column). p0 and j are the
# generalised least squares solution over the free alignment parameters,
#     min sum_c r_c^T Sigma_c^-1 r_c, r_c = p_c - p0 - j (y_c - <y>).
# The orbit shift at the reference of the remainder r_c is S(G) r_c,
# S(G) the orbit response to the alignment at the gradient setting G.

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
    with the reference at the Al foil and no steering response

    plane: "Long" (y) or "Short" (x)

    return: beamline
    """
    quadSigmaConstraintShift = np.inf
    quadSigmaConstraintTilt = np.inf

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
    return beamline

# --------------------------------------------------------------
def fitSteeringResponse(
        positions: npt.NDArray[np.float64],
        parsList: list[npt.NDArray[np.float64]],
        covList: list[npt.NDArray[np.float64]],
        free: npt.NDArray[np.bool_]) -> dict:
    """
    Generalised least squares fit of the condition alignments
    linear in the condition beam position,
        p_c = p0 + j (y_c - <y>)

    positions: mean beam position at BPM0 per condition [m]
    parsList: alignment parameters per condition [m, rad]
    covList: alignment covariance per condition [m, rad]
    free: free alignment parameters

    return: dict with p0 and j (zero for the fixed parameters),
            their covariance (free parameters, [p0, j]),
            the remainders r_c, chi2 and ndf
    """
    freeIdx = np.where(free)[0]
    nFree = len(freeIdx)
    nConditions = len(positions)
    meanPosition = np.mean(positions)

    N = np.zeros(shape=(2 * nFree, 2 * nFree), dtype=np.float64)
    h = np.zeros(shape=(2 * nFree), dtype=np.float64)
    for c in range(nConditions):
        X = np.hstack([np.eye(nFree), (positions[c] - meanPosition) * np.eye(nFree)])
        W = np.linalg.inv(covList[c][np.ix_(freeIdx, freeIdx)])
        N += X.T @ W @ X
        h += X.T @ W @ parsList[c][freeIdx]

    cov = np.linalg.inv(N)
    sol = cov @ h

    p0 = np.zeros(shape=(len(free)), dtype=np.float64)
    j = np.zeros(shape=(len(free)), dtype=np.float64)
    p0[freeIdx] = sol[0:nFree]
    j[freeIdx] = sol[nFree:2 * nFree]

    remainders = []
    chi2 = 0.0
    for c in range(nConditions):
        r = parsList[c] - p0 - j * (positions[c] - meanPosition)
        W = np.linalg.inv(covList[c][np.ix_(freeIdx, freeIdx)])
        chi2 += r[freeIdx] @ W @ r[freeIdx]
        remainders.append(r)

    return {
        "p0": p0,
        "j": j,
        "meanPosition": meanPosition,
        "cov": cov,
        "remainders": remainders,
        "chi2": chi2,
        "ndf": nFree * (nConditions - 2)}

# --------------------------------------------------------------
def estimateSteeringResponse(
        beamline: E320Beamline,
        conditions: list[npt.NDArray[np.int64]],
        bpmShots: npt.NDArray[np.float64],
        bpmErrs: npt.NDArray[np.float64],
        quadGrads: npt.NDArray[np.float64],
        gradientResolution: float) -> dict:
    """
    Steering response J of one plane from the steering conditions

    beamline: beamline of the plane, no steering response,
        the alignment parameters are the starting point of the fits
    conditions: shot indices of the steering conditions
    bpmShots: bpm measurements [m]
    bpmErrs: bpm measurement errors [m]
    quadGrads: quadrupole gradients [T/m]
    gradientResolution: gradient quantisation of the settings [T/m]

    return: dict with the 10x2 response J, the linear fit
            and the per condition results
    """
    iniPars = beamline.getAlignmentPars()
    free = np.concatenate([beamline.bpmFree, beamline.quadShiftsFree, beamline.quadTiltsFree])

    # Alignment and mean beam vector at BPM0 per condition
    positions = []
    angles = []
    parsList = []
    covList = []
    for c, shots in enumerate(conditions):
        print("-"*50)
        print(f"Condition {c}: {len(shots)} shots")
        beamline.setAlignmentPars(iniPars)
        pars, cov, chi2, ndf = beamline.optimizeBeamline(bpmShots[shots], bpmErrs[shots], quadGrads[shots])
        if (ndf < 0):
            print(f"Condition {c}: alignment fit failed, skipping")
            continue

        beamline.setAlignmentPars(pars)
        beamVecs = np.zeros(shape=(len(shots), 2), dtype=np.float64)
        for k, idx in enumerate(shots):
            beamVecs[k] = beamline.fitBeamNewtonStep(bpmShots[idx], bpmErrs[idx], quadGrads[idx])[0]

        positions.append(np.mean(beamVecs[:,0]))
        angles.append(np.mean(beamVecs[:,1]))
        parsList.append(pars)
        covList.append(cov)
        print(f"Condition {c}: beam at BPM0 ({positions[-1] * 1e3:.3f} mm, {angles[-1] * 1e3:.3f} mrad), chi2/ndf {chi2 / ndf:.2f}")
    beamline.setAlignmentPars(iniPars)

    positions = np.array(positions)
    angles = np.array(angles)
    if (len(positions) < 3):
        print("At least 3 steering conditions are needed")
        return {}

    fit = fitSteeringResponse(positions, parsList, covList, free)

    response = np.zeros(shape=(10, 2), dtype=np.float64)
    response[:,0] = fit["j"]

    # Direction of the scan in (x, x') at BPM0
    scanSlope, scanOffset = np.polyfit(positions, angles, 1)
    scanSpread = np.std(angles - (scanSlope * positions + scanOffset))

    # Orbit shifts at the reference per gradient setting: of the
    # condition alignments about their mean, and of the linear fit remainders
    settings = defaultdict(list)
    for idx in range(len(quadGrads)):
        settings[tuple(np.round(quadGrads[idx] / gradientResolution).astype(int))].append(idx)

    meanPars = np.mean(np.array(parsList), axis=0)
    shiftsRms = []
    remaindersRms = []
    for settingKey, shots in settings.items():
        S = beamline.getOrbitResponseMatrix(bpmErrs[shots[0]], np.mean(quadGrads[shots], axis=0))
        shifts = np.array([S @ (pars - meanPars) for pars in parsList])
        remainders = np.array([S @ r for r in fit["remainders"]])
        shiftsRms.append(np.sqrt(np.mean(shifts**2, axis=0)))
        remaindersRms.append(np.sqrt(np.mean(remainders**2, axis=0)))
    shiftsRms = np.array(shiftsRms)
    remaindersRms = np.array(remaindersRms)

    print("-"*50)
    print(f"{len(positions)} conditions, beam position at BPM0 {np.min(positions) * 1e3:.3f} to {np.max(positions) * 1e3:.3f} mm")
    print(f"Scan direction: angle = {scanSlope:.3f} mrad/mm * position + const, spread {scanSpread * 1e6:.2f} urad "
          "(J is assigned to the position only)")
    print(f"Linear fit: chi2/ndf = {fit['chi2']:.1f}/{fit['ndf']}")
    jErrs = np.zeros(shape=(10), dtype=np.float64)
    freeIdx = np.where(free)[0]
    jErrs[freeIdx] = np.sqrt(np.linalg.diagonal(fit["cov"])[len(freeIdx):])
    for i in range(10):
        if (free[i]):
            print(f"  J[{i}, 0] = {fit['j'][i]:.5e} +- {jErrs[i]:.1e}")
    print(f"Orbit shift at the reference, RMS over the conditions, median (max) over {len(settings)} settings [um, urad]:")
    print(f"  condition alignments about their mean: "
          f"{np.median(shiftsRms[:,0]) * 1e6:.2f}, {np.median(shiftsRms[:,1]) * 1e6:.2f} "
          f"({np.max(shiftsRms[:,0]) * 1e6:.2f}, {np.max(shiftsRms[:,1]) * 1e6:.2f})")
    print(f"  remainder of the linear fit: "
          f"{np.median(remaindersRms[:,0]) * 1e6:.2f}, {np.median(remaindersRms[:,1]) * 1e6:.2f} "
          f"({np.max(remaindersRms[:,0]) * 1e6:.2f}, {np.max(remaindersRms[:,1]) * 1e6:.2f})")

    return {
        "response": response,
        "responseErrs": jErrs,
        "fit": fit,
        "positions": positions,
        "angles": angles,
        "pars": parsList,
        "covs": covList,
        "shiftsRms": shiftsRms,
        "remaindersRms": remaindersRms}

# --------------------------------------------------------------
def processRun(
        pathToInFile: str,
        plane: str,
        gradientResolution: float) -> dict:
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
    result = estimateSteeringResponse(beamline, conditions, bpmShots, bpmErrs, quadGrads, gradientResolution)
    if (len(result) == 0):
        return result

    # ------------------------------------------------------
    labels = [
        "bpm0 shift", "bpm1 shift", "bpm2 shift", "bpm3 shift",
        "quad1 shift", "quad2 shift", "quad3 shift",
        "quad1 tilt", "quad2 tilt", "quad3 tilt"]
    print("-"*50)
    print("Steering response for the constraints file:")
    print(f"    steeringResponse{plane} = [")
    for i in range(10):
        print(f"        [{result['response'][i,0]:.6e}, {result['response'][i,1]:.1f}], # {labels[i]}")
    print("    ]")

    return result

if __name__ == "__main__":
    pathToInFile = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                    "quad_scan_data_facet/sm_quad_scan_steering/outFile.csv")

    processRun(pathToInFile = pathToInFile,
               plane = "Long",
               gradientResolution = 1e-3)
