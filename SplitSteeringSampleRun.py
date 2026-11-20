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
def processRun(pathToInFile: str) -> None:
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

        print(c)
        print(shots)
        filePath = "condition-" + str(c) + ".csv"
        outFile = open(filePath, "w", newline="")
        writer = csv.writer(outFile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for shot in shots:
            writer.writerow([
                bpmXs[shot, 0], 
                bpmXs[shot, 1], 
                bpmXs[shot, 2], 
                bpmXs[shot, 3], 
                bpmYs[shot, 0], 
                bpmYs[shot, 1], 
                bpmYs[shot, 2], 
                bpmYs[shot, 3], 
                quadGrads[shot, 0], 
                quadGrads[shot, 1], 
                quadGrads[shot, 2]])
        
if __name__ == "__main__":
    pathToInFile = ("/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/"
                    "quad_scan_data_facet/sm_quad_scan_steering/outFile.csv")
    processRun(pathToInFile = pathToInFile)
