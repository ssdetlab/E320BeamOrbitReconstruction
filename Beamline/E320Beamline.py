from Geometry.GeneralConstraints import GeneralConstraints
from Beamline.FreePropagation import FreePropagation
from Types.BeamOrbitFitResult import BeamOrbitFitResult
from Types.BeamState import BeamState

import numpy as np
import numpy.typing as npt
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from abc import ABC, abstractmethod

def printMatrix(a):
    for row in a:
        for col in row:
            print("{:8.6f}".format(col), end=" ")
        print("")

class E320Beamline(ABC):
    """
    Base E320 beamline class
    """
    def __init__(self): 
        """
        Constructor
        """
        # Number of processes for the alignment fit
        # (None -> number of CPUs)
        self.nProcesses = None

        # Beam energy [J] -> 10 GeV
        self.energy = 10 * 1e9 * GeneralConstraints.e

        # Quad shift [m]
        self.quadShift = np.zeros(shape=(3), dtype=float)  
        self.quadShift[0] = 0.0 # quad 0
        self.quadShift[1] = 0.0 # quad 1
        self.quadShift[2] = 0.0 # quad 2

        # Quad tilt [rad]
        self.quadTilt = np.zeros(shape=(3), dtype=float)  
        self.quadTilt[0] = 0.0 # quad 0
        self.quadTilt[1] = 0.0 # quad 1
        self.quadTilt[2] = 0.0 # quad 2

        # Quad length [m]
        self.quadLength = np.zeros(shape=(3), dtype=float)   
        self.quadLength[0] = None
        self.quadLength[1] = None
        self.quadLength[2] = None

        # BPM shift [m]
        self.bpmShift = np.zeros(shape=(4), dtype=float)    
        self.bpmShift[0] = 0.0 # before the IP
        self.bpmShift[1] = 0.0 # after quad 0
        self.bpmShift[2] = 0.0 # after quad 1
        self.bpmShift[3] = 0.0 # after quad 2

        # Distance from free_space(i) to Quad(i+1)
        self.LQ = np.zeros(shape=(3), dtype=float)  
        self.LQ[0] = None
        self.LQ[1] = None
        self.LQ[2] = None

        # Distance from Quad(i) to BPM(i)
        self.LB = np.zeros(shape=(3), dtype=float)  
        self.LB[0] = None
        self.LB[1] = None
        self.LB[2] = None

        # Distance from the reference point to BPM0
        self.bpm0CenterPrimary = 0
        self.referenceDist = 0
        self.toReference = FreePropagation(self.referenceDist)

        # Beam (x, x') covariance at the reference within one bunch [m, rad]
        self.beamCov = np.zeros(shape=(2, 2), dtype=float)

        # Alignment fit covariance in the getAlignmentPars order [m, rad]
        # (zero -> no alignment contribution to the orbit covariance)
        self.alignmentCov = np.zeros(shape=(10, 10), dtype=float)

        # Alignment parameters of the alignment samples 
        # in the getAlignmentPars order [m, rad]
        # (empty -> no alignment sample contribution)
        self.alignmentSamples = []

        # Free parameters of the alignment fit
        # (fixed BPM0 and BPM1 define the gauge: translation + tilt)
        self.bpmFree = np.zeros(shape=(4), dtype=bool)
        self.bpmFree[0] = False # before the IP
        self.bpmFree[1] = False # after quad 0
        self.bpmFree[2] = True # after quad 1
        self.bpmFree[3] = True # after quad 2

        self.quadShiftsFree = np.zeros(shape=(3), dtype=bool)
        self.quadShiftsFree[0] = True # quad 0
        self.quadShiftsFree[1] = True # quad 1
        self.quadShiftsFree[2] = True # quad 2

        self.quadTiltsFree = np.zeros(shape=(3), dtype=bool)
        self.quadTiltsFree[0] = False # quad 0
        self.quadTiltsFree[1] = False # quad 1
        self.quadTiltsFree[2] = False # quad 2

        # Gaussian constraints on the BPM shifts [m]
        # (sigma = np.inf -> unconstrained)
        self.bpmConstraintMean = np.zeros(shape=(4), dtype=float)
        self.bpmConstraintSigma = np.zeros(shape=(4), dtype=float)
        self.bpmConstraintSigma[0] = np.inf
        self.bpmConstraintSigma[1] = np.inf
        self.bpmConstraintSigma[2] = np.inf
        self.bpmConstraintSigma[3] = np.inf

        # Gaussian constraints on the quad shifts [m]
        # (sigma = np.inf -> unconstrained)
        self.quadShiftsConstraintMean = np.zeros(shape=(3), dtype=float)
        self.quadShiftsConstraintSigma = np.zeros(shape=(3), dtype=float)
        self.quadShiftsConstraintSigma[0] = np.inf
        self.quadShiftsConstraintSigma[1] = np.inf
        self.quadShiftsConstraintSigma[2] = np.inf

        # Gaussian constraints on the quad tilts [rad]
        # (sigma = np.inf -> unconstrained)
        self.quadTiltsConstraintMean = np.zeros(shape=(3), dtype=float)
        self.quadTiltsConstraintSigma = np.zeros(shape=(3), dtype=float)
        self.quadTiltsConstraintSigma[0] = np.inf
        self.quadTiltsConstraintSigma[1] = np.inf
        self.quadTiltsConstraintSigma[2] = np.inf

    def setBeamEnergy(self, energy: float) -> None:
        """
        Set beam energy

        energy: beam energy [J]
        """
        self.energy = energy

    def setQuadShifts(self, shifts: npt.NDArray[np.float64]) -> None:
        """
        Set quadrupole shifts

        shifts: quadrupole shifts [m]
        """
        self.quadShift[0] = shifts[0]
        self.quadShift[1] = shifts[1]
        self.quadShift[2] = shifts[2]

    def setQuadTilts(self, tilts: npt.NDArray[np.float64]) -> None:
        """
        Set quadrupole tilts

        shifts: quadrupole tilts [rad]
        """
        self.quadTilt[0] = tilts[0]
        self.quadTilt[1] = tilts[1]
        self.quadTilt[2] = tilts[2]

    def setQuadLengths(self, lengths: npt.NDArray[np.float64]) -> None:
        """
        Set quadrupole lengths

        lengths: quadrupole lengths [m]
        """
        self.quadLength[0] = lengths[0]
        self.quadLength[1] = lengths[1]
        self.quadLength[2] = lengths[2]

    def setBpmShifts(self, shifts: npt.NDArray[np.float64]) -> None:
        """
        Set BPM shifts

        shifts: BPM shifts [m]
        """
        self.bpmShift[0] = shifts[0]
        self.bpmShift[1] = shifts[1]
        self.bpmShift[2] = shifts[2]
        self.bpmShift[3] = shifts[3]

    def setFreeSpaceQuadDistances(self, distances: npt.NDArray[np.float64]) -> None:
        """
        Set distances between the free_space(i) and quad(i+1)

        distances: distances [m]
        """
        self.LQ[0] = distances[0]
        self.LQ[1] = distances[1]
        self.LQ[2] = distances[2]   

    def setQuadBPMDistances(self, distances: npt.NDArray[np.float64]) -> None:
        """
        Set distances between the quad(i) and bpm(i+1)

        distances: distances [m]
        """
        self.LB[0] = distances[0]
        self.LB[1] = distances[1]
        self.LB[2] = distances[2]

    def setReferenceDistance(self, bpm0Center, distance: float) -> None:
        """
        Set the distance from the fit starting surface to 
        reference surface

        bpm0Center: center of BPM0 along primary axis [m]
        distances: distance [m]
        """
        
        self.bpm0CenterPrimary = bpm0Center
        self.referenceDist = distance
        self.toReference = FreePropagation(distance)

    def setBeamCov(self, size: float, divergence: float, correlation: float) -> None:
        """
        Set the (x, x') covariance of the beam at the reference 
        within one bunch

        size: beam size [m]
        divergence: beam divergence [rad]
        correlation: size-divergence correlation coefficient
        """
        self.beamCov[0,0] = size**2
        self.beamCov[0,1] = correlation * size * divergence
        self.beamCov[1,0] = correlation * size * divergence
        self.beamCov[1,1] = divergence**2

    def setAlignmentCov(self, cov: npt.NDArray[np.float64]) -> None:
        """
        Set the alignment fit covariance propagated to the orbit 
        at the reference by optimizeBeam

        cov: 10x10 covariance in the getAlignmentPars order [m, rad]
        """
        for i in range(10):
            for j in range(10):
                self.alignmentCov[i,j] = cov[i,j]

    def setAlignmentSamples(self, samples: list[npt.NDArray[np.float64]]) -> None:
        """
        Set the alignment parameters of the alignment samples, 
        their orbit shifts with respect to the current alignment 
        at the reference are computed by optimizeBeam

        samples: list of alignment parameters in the getAlignmentPars order [m, rad]
        """
        self.alignmentSamples = []
        for sample in samples:
            self.alignmentSamples.append(np.array(sample, dtype=float))

    def setBpmFree(self, free: npt.NDArray[np.bool_]) -> None:
        """
        Set which BPM shifts are free in the alignment fit

        free: True -> free, False -> fixed to the current shift
        """
        self.bpmFree[0] = free[0]
        self.bpmFree[1] = free[1]
        self.bpmFree[2] = free[2]
        self.bpmFree[3] = free[3]

    def setQuadShiftsFree(self, free: npt.NDArray[np.bool_]) -> None:
        """
        Set which quad shifts are free in the alignment fit

        free: True -> free, False -> fixed to the current shift
        """
        self.quadShiftsFree[0] = free[0]
        self.quadShiftsFree[1] = free[1]
        self.quadShiftsFree[2] = free[2]

    def setQuadTiltsFree(self, free: npt.NDArray[np.bool_]) -> None:
        """
        Set which quad tilts are free in the alignment fit

        free: True -> free, False -> fixed to the current tilt
        """
        self.quadTiltsFree[0] = free[0]
        self.quadTiltsFree[1] = free[1]
        self.quadTiltsFree[2] = free[2]

    def setBpmConstraints(self, means: npt.NDArray[np.float64], sigmas: npt.NDArray[np.float64]) -> None:
        """
        Set Gaussian constraints on the BPM shifts

        means: constraint central values [m]
        sigmas: constraint widths [m], np.inf -> unconstrained
        """
        self.bpmConstraintMean[0] = means[0]
        self.bpmConstraintMean[1] = means[1]
        self.bpmConstraintMean[2] = means[2]
        self.bpmConstraintMean[3] = means[3]

        self.bpmConstraintSigma[0] = sigmas[0]
        self.bpmConstraintSigma[1] = sigmas[1]
        self.bpmConstraintSigma[2] = sigmas[2]
        self.bpmConstraintSigma[3] = sigmas[3]

    def setQuadShiftsConstraints(self, means: npt.NDArray[np.float64], sigmas: npt.NDArray[np.float64]) -> None:
        """
        Set Gaussian constraints on the quad shifts

        means: constraint central values [m]
        sigmas: constraint widths [m], np.inf -> unconstrained
        """
        self.quadShiftsConstraintMean[0] = means[0]
        self.quadShiftsConstraintMean[1] = means[1]
        self.quadShiftsConstraintMean[2] = means[2]

        self.quadShiftsConstraintSigma[0] = sigmas[0]
        self.quadShiftsConstraintSigma[1] = sigmas[1]
        self.quadShiftsConstraintSigma[2] = sigmas[2]
    
    def setQuadTiltsConstraints(self, means: npt.NDArray[np.float64], sigmas: npt.NDArray[np.float64]) -> None:
        """
        Set Gaussian constraints on the quad tilts

        means: constraint central values [rad]
        sigmas: constraint widths [rad], np.inf -> unconstrained
        """
        self.quadTiltsConstraintMean[0] = means[0]
        self.quadTiltsConstraintMean[1] = means[1]
        self.quadTiltsConstraintMean[2] = means[2]

        self.quadTiltsConstraintSigma[0] = sigmas[0]
        self.quadTiltsConstraintSigma[1] = sigmas[1]
        self.quadTiltsConstraintSigma[2] = sigmas[2]
    
    def setNumProcesses(self, nProcesses: int) -> None:
        """
        Set the number of processes used in the alignment fit

        nProcesses: number of processes, None -> number of CPUs
        """
        self.nProcesses = nProcesses

    def getNumProcesses(self) -> int:
        """
        Get the number of processes used in the alignment fit

        return: number of processes
        """
        if (self.nProcesses is None):
            return mp.cpu_count()
        return self.nProcesses

    def getAlignmentPars(self) -> npt.NDArray[np.float64]:
        """
        Get alignment parameters

        return: [bpm0, bpm1, bpm2, bpm3, quad0, quad1, quad2] shifts [m]
        """
        return np.array([
            self.bpmShift[0], 
            self.bpmShift[1], 
            self.bpmShift[2], 
            self.bpmShift[3],
            self.quadShift[0], 
            self.quadShift[1], 
            self.quadShift[2],
            self.quadTilt[0], 
            self.quadTilt[1], 
            self.quadTilt[2]], dtype=float)

    def setAlignmentPars(self, pars: npt.NDArray[np.float64]) -> None:
        """
        Set alignment parameters

        pars: [bpm0, bpm1, bpm2, bpm3] shifts [m] X [quad0, quad1, quad2] shifts [m] X [quad0, quad1, quad2] tilts [rad]
        """
        self.setBpmShifts(pars[0:4])
        self.setQuadShifts(pars[4:7])
        self.setQuadTilts(pars[7:10])

    def getPars(self):
        """
        Get beamline parameters

        return: list of beamline parameters
        """
        return [self.quadShift, self.bpmShift, (self.energy / GeneralConstraints.e) / 1e9, self.quadLength, self.LQ, self.LB]

    @abstractmethod
    def getBpms(self, start, quadGrads) -> list[float]:
        return []

    @abstractmethod
    def getBeamStates(self, bpmMeas, bpmErrs, vec, cov, quadGrads) -> list[BeamState]:
        return []

    def getResponseBase(self, grads: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        BPM prediction for zero beam vector (x, x') at BPM0
        at the current alignment parameters

        The BPM prediction is affine in the beam vector
        and in the alignment parameters:
            bpms = base + A (x, x') + B dPars

        grads: quadrupole gradients [T/m]

        return: base [m]
        """
        return np.array(self.getBpms([0.0, 0.0], grads))

    def getBeamResponseMatrix(self, base: npt.NDArray[np.float64], grads: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Response of the BPM prediction to the beam vector (x, x') at BPM0
        (exact, the prediction is affine in the beam vector)

        base: getResponseBase at the same alignment parameters and gradients [m]
        grads: quadrupole gradients [T/m]

        return: 4x2 matrix A [m/m, m/rad]
        """
        beamDer = np.zeros(shape=(4, 2), dtype=float)
        beamDer[:,0] = np.array(self.getBpms([1.0, 0.0], grads)) - base
        beamDer[:,1] = np.array(self.getBpms([0.0, 1.0], grads)) - base
        return beamDer

    def getAlignmentResponseMatrix(self, base: npt.NDArray[np.float64], grads: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Response of the BPM prediction to the alignment parameters
        (exact, the prediction is affine in the alignment parameters),
        the alignment parameters are restored on return

        base: getResponseBase at the same alignment parameters and gradients [m]
        grads: quadrupole gradients [T/m]

        return: 4x10 matrix B, columns in the getAlignmentPars order
            [bpm0, bpm1, bpm2, bpm3, quad0, quad1, quad2] shifts [m/m] X
            [quad0, quad1, quad2] tilts [m/rad]
        """
        currentPars = self.getAlignmentPars()
        alignDer = np.zeros(shape=(4, len(currentPars)), dtype=float)
        for i in range(len(currentPars)):
            shift = np.zeros_like(currentPars)
            shift[i] += 1
            self.setAlignmentPars(currentPars + shift)
            alignDer[:,i] = np.array(self.getBpms([0.0, 0.0], grads)) - base
            self.setAlignmentPars(currentPars)
        return alignDer

    def getBpmWeights(self, bpmErrs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        BPM weights of the beam vector fit

        bpmErrs: bpm measurement errors [m]

        return: 4x4 diagonal weight matrix W [1/m^2]
        """
        weights = np.zeros(shape=(4, 4), dtype=float)
        weights[0,0] = 1.0 / bpmErrs[0]**2
        weights[1,1] = 1.0 / bpmErrs[1]**2
        weights[2,2] = 1.0 / bpmErrs[2]**2
        weights[3,3] = 1.0 / bpmErrs[3]**2
        return weights

    def getOrbitResponseMatrix(self, bpmErrs: npt.NDArray[np.float64], grads: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Response of the fitted beam vector (x, x') at the reference
        to the alignment parameters (exact, the fitted beam vector
        is linear in the alignment parameters):
            S = -T (A^T W A)^-1 A^T W B
        with T the BPM0 to reference propagation.
        The orbit covariance due to the alignment covariance cov
        is S cov S^T, the orbit shift due to the change dPars is S dPars

        bpmErrs: bpm measurement errors [m]
        grads: quadrupole gradients [T/m]

        return: 2x10 matrix S, columns in the getAlignmentPars order,
            rows (x, x') [m/m, m/rad; rad/m, rad/rad]
        """
        base = self.getResponseBase(grads)
        beamDer = self.getBeamResponseMatrix(base, grads)
        alignDer = self.getAlignmentResponseMatrix(base, grads)
        weights = self.getBpmWeights(bpmErrs)

        wBeamDer = weights @ beamDer
        beamCov = np.linalg.inv(beamDer.T @ wBeamDer)

        return -self.toReference.ABCD @ beamCov @ wBeamDer.T @ alignDer

    def fitBeamNewtonStep(self, bpmMeas, bpmErrs, grads):
        """
        Fit of the beam vector (x, x') at BPM0 for a single shot

        The BPM prediction is affine in the beam vector, 
        bpms = base + A (x, x'), so the chi2 is quadratic and 
        a single Newton step from (0, 0) gives the exact minimum:
            (x, x') = (A^T W A)^-1 A^T W (bpmMeas - base)
        with the covariance (A^T W A)^-1

        bpmMeas: bpm measurements [m]
        bpmErrs: bpm measurement errors [m]
        grads: quadrupole gradients [T/m]

        return: beam (x, x') at BPM0, its covariance, 
            beam positions at the bpms, chi2
        """
        # Prediction for zero beam vector
        base = self.getResponseBase(grads)

        # Response to the beam vector (x, x') at BPM0
        beamDer = self.getBeamResponseMatrix(base, grads)

        # BPM weights
        weights = self.getBpmWeights(bpmErrs)

        # Newton step
        res = np.array(bpmMeas) - base
        wBeamDer = weights @ beamDer
        beamCov = np.linalg.inv(beamDer.T @ wBeamDer)
        beamVec = beamCov @ wBeamDer.T @ res

        bpms = base + beamDer @ beamVec
        resFit = np.array(bpmMeas) - bpms
        chi2 = resFit @ weights @ resFit

        return (beamVec, beamCov, chi2)

    def optimizeBeam(self, bpmMeas, bpmErrs, grads): 
        """
        Run beam orbit fit

        bpmMeas: list of bpm measurements [m]
        bpmErrs: list of bpm measurement errors [m]
        grads: list of quadrupole gradients [T/m]

        return: beam (x, x') at the reference, the list of beam positions at the bpms
            and the (x, x') covariance at the reference
        """
        beamVec, beamCov, chi2 = self.fitBeamNewtonStep(bpmMeas, bpmErrs, grads)

        vec = self.toReference.propagate(beamVec[None,:])[0]
        cov = self.toReference.ABCD @ beamCov @ self.toReference.ABCD.T
        states = self.getBeamStates(bpmMeas, bpmErrs, beamVec, beamCov, grads)

        # Response of the orbit at the reference to the alignment parameters
        nSamples = len(self.alignmentSamples)
        if (np.any(self.alignmentCov != 0) or nSamples > 0):
            orbitDer = self.getOrbitResponseMatrix(bpmErrs, grads)
        else: 
            orbitDer = np.array(["None"])

        # Orbit covariance at the reference due to the alignment uncertainty, 
        # common to all events
        alignmentOrbitCov = np.zeros(shape=(2, 2), dtype=float)
        if (np.any(self.alignmentCov != 0)):
            alignmentOrbitCov = orbitDer @ self.alignmentCov @ orbitDer.T

        # Orbit shifts at the reference for the alignment samples 
        # with respect to the current alignment, common to all events, 
        # and their covariance diag(half-range^2)
        alignmentSampleShifts = np.zeros(shape=(nSamples, 2), dtype=float)
        alignmentSampleCov = np.zeros(shape=(2, 2), dtype=float)
        if (nSamples > 0):
            pars = self.getAlignmentPars()
            for k in range(nSamples):
                alignmentSampleShifts[k] = orbitDer @ (self.alignmentSamples[k] - pars)
            halfRange = 0.5 * (np.max(alignmentSampleShifts, axis=0) - np.min(alignmentSampleShifts, axis=0))
            alignmentSampleCov[0,0] = halfRange[0]**2
            alignmentSampleCov[1,1] = halfRange[1]**2

        return BeamOrbitFitResult(
                states, 
                vec, 
                cov, 
                self.bpm0CenterPrimary + self.referenceDist, 
                self.energy, 
                chi2, 
                np.array(self.beamCov), 
                alignmentOrbitCov, 
                alignmentSampleShifts, 
                alignmentSampleCov)

    def getBeamlineChi2Block(self, bpmShots, bpmErrs, quadGrads):
        """
        Sum of the shot chi2 for a block of shots 
        at the current alignment parameters, with the beam 
        vector (x, x') at BPM0 of each shot profiled out
        by fitBeamNewtonStep

        bpmShots: Nx4 array of bpm measurements [m]
        bpmErrs: Nx4 array of bpm measurement errors [m]
        quadGrads: Nx3 array of quadrupole gradients [T/m]

        return: chi2
        """
        chi2 = 0.0
        nShots = bpmShots.shape[0]
        for idx in range(nShots):
            beamVec, beamCov, shotChi2 = self.fitBeamNewtonStep(
                    bpmShots[idx,:], 
                    bpmErrs[idx,:], 
                    quadGrads[idx,:])
            chi2 += shotChi2
        return chi2

    def getBeamlineChi2(self, pars, bpmShots, bpmErrs, quadGrads, pool=None):
        """
        Data chi2 for given alignment parameters with the beam 
        vector (x, x') at BPM0 of each shot profiled out
        by fitBeamNewtonStep

        pars: alignment parameters [bpm0, bpm1, bpm2, bpm3, quad0, quad1, quad2] [m]
        bpmShots: Nx4 array of bpm measurements [m]
        bpmErrs: Nx4 array of bpm measurement errors [m]
        quadGrads: Nx3 array of quadrupole gradients [T/m]
        pool: multiprocessing Pool, None -> serial evaluation

        return: chi2
        """
        parsOld = self.getAlignmentPars()
        self.setAlignmentPars(pars)

        if (pool is None):
            chi2 = self.getBeamlineChi2Block(bpmShots, bpmErrs, quadGrads)
        else:
            # Split the shots into blocks, a few per process
            # for load balancing
            nShots = bpmShots.shape[0]
            nBlocks = 4 * self.getNumProcesses()
            blockSize = nShots // nBlocks + 1

            args = []
            for iStart in range(0, nShots, blockSize):
                iStop = iStart + blockSize
                args.append((
                    bpmShots[iStart:iStop,:], 
                    bpmErrs[iStart:iStop,:], 
                    quadGrads[iStart:iStop,:]))

            # The alignment parameters are sent to 
            # the processes together with self
            chi2 = np.sum(pool.starmap(self.getBeamlineChi2Block, args))

        self.setAlignmentPars(parsOld)
        return chi2

    def optimizeBeamline(self, bpmShots, bpmErrs, quadGrads, starts=None):
        """
        Run constrained chi2 fit of the BPM and quad shifts

        Free parameters: setBpmFree, setQuadShiftsFree, setQuadTiltsFree
        (at least two parameters have to be fixed to remove the
        translation and tilt degeneracies)
        Gaussian constraints: setBpmConstraints, setQuadConstraints
        Number of processes: setNumProcesses

        bpmShots: Nx4 array of bpm measurements [m]
        bpmErrs: Nx4 array of bpm measurement errors [m]
        quadGrads: Nx3 array of quadrupole gradients [T/m]
        starts: list of starting points [bpm0, bpm1, bpm2, bpm3, quad0, quad1, quad2] [m],
            only the free entries are used; None -> current shifts

        return: alignment parameters [bpm0, bpm1, bpm2, bpm3, quad0, quad1, quad2] [m],
            their covariance (zero for the fixed parameters) [m^2], 
            data chi2, number of degrees of freedom
        """
        
        print("-"*50)
        nPars = 10

        # The minimizer works in [mm] to keep the parameters O(1)
        parScale = 1e-3

        parsIni = self.getAlignmentPars()

        free = np.array([
            self.bpmFree[0], 
            self.bpmFree[1], 
            self.bpmFree[2], 
            self.bpmFree[3],
            self.quadShiftsFree[0], 
            self.quadShiftsFree[1], 
            self.quadShiftsFree[2], 
            self.quadTiltsFree[0], 
            self.quadTiltsFree[1], 
            self.quadTiltsFree[2]], dtype=bool)
        constraintMean = np.array([
            self.bpmConstraintMean[0], 
            self.bpmConstraintMean[1], 
            self.bpmConstraintMean[2], 
            self.bpmConstraintMean[3],
            self.quadShiftsConstraintMean[0], 
            self.quadShiftsConstraintMean[1], 
            self.quadShiftsConstraintMean[2], 
            self.quadTiltsConstraintMean[0], 
            self.quadTiltsConstraintMean[1], 
            self.quadTiltsConstraintMean[2]], dtype=float)
        constraintSigma = np.array([
            self.bpmConstraintSigma[0], 
            self.bpmConstraintSigma[1], 
            self.bpmConstraintSigma[2], 
            self.bpmConstraintSigma[3],
            self.quadShiftsConstraintSigma[0], 
            self.quadShiftsConstraintSigma[1], 
            self.quadShiftsConstraintSigma[2],
            self.quadTiltsConstraintSigma[0], 
            self.quadTiltsConstraintSigma[1], 
            self.quadTiltsConstraintSigma[2]], dtype=float)

        freeIdx = []
        for iPar in range(nPars):
            if (free[iPar] == True):
                freeIdx.append(iPar)
        nFree = len(freeIdx)

        def getFullPars(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            pars = np.array(parsIni)
            for i in range(nFree):
                pars[freeIdx[i]] = x[i] * parScale
            return pars

        def objective(x: npt.NDArray[np.float64]) -> float:
            pars = getFullPars(x)
            chi2 = self.getBeamlineChi2(pars, bpmShots, bpmErrs, quadGrads, pool)
            # Gaussian constraints, sigma = np.inf -> no contribution
            for iPar in freeIdx:
                chi2 += ((pars[iPar] - constraintMean[iPar]) / constraintSigma[iPar])**2
            return float(chi2)

        def gradient(x: npt.NDArray[np.float64]):
            return approx_derivative(
                objective, x, method="3-point"
            ).ravel()

        with mp.Pool(processes=self.getNumProcesses()) as pool:
            if (starts is None):
                starts = [parsIni]

            bestResult = None
            for start in starts:
                xIni = np.zeros(shape=(nFree), dtype=float)
                for i in range(nFree):
                    xIni[i] = start[freeIdx[i]] / parScale

                result = minimize(
                    objective, 
                    xIni,
                    jac=gradient,
                    method='SLSQP', 
                    tol=1e-6)
                print(f"Start {xIni} [mm, mrad]: success {result.success}, chi2 {result.fun}, {result.message}")

                if (result.success == True):
                    if (bestResult is None or result.fun < bestResult.fun):
                        bestResult = result

            if (bestResult is None):
                print("Error, optimizeBeamline: no successful minimization")
                return (np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), np.zeros(shape=(nPars, nPars), dtype=float), -1, -1)

            # Covariance from the Hessian of the chi2
            H = approx_derivative(
                gradient, bestResult.x, method="3-point"
            )
            H = np.array(0.5 * (H + H.T))
            covFree = 2.0 * np.linalg.inv(H) * parScale**2

            pars = getFullPars(bestResult.x)
            cov = np.zeros(shape=(nPars, nPars), dtype=float)
            for i in range(nFree):
                for j in range(nFree):
                    cov[freeIdx[i], freeIdx[j]] = covFree[i,j]

            chi2 = self.getBeamlineChi2(pars, bpmShots, bpmErrs, quadGrads, pool)

        ndf = 2 * bpmShots.shape[0] - nFree
        chi2Constraints = bestResult.fun - chi2

        self.setAlignmentPars(pars)

        parsErr = np.zeros(shape=(nPars), dtype=float)
        for iPar in range(nPars):
            parsErr[iPar] = np.sqrt(cov[iPar,iPar])

        print(f"Alignment fit, {bpmShots.shape[0]} shots, {nFree} free parameters")
        print(f"BPM shifts [mm]: {pars[0:4] * 1e3}")
        print(f"BPM shift errors [mm]: {parsErr[0:4] * 1e3}")
        print(f"Quad shifts [mm]: {pars[4:7] * 1e3}")
        print(f"Quad shift errors [mm]: {parsErr[4:7] * 1e3}")
        print(f"Quad tilts [mrad]: {pars[7:10] * 1e3}")
        print(f"Quad tilt errors [mrad]: {parsErr[7:10] * 1e3}")
        print(f"chi2/ndf (data): {chi2} / {ndf} = {chi2 / ndf}")
        print(f"chi2 (constraints): {chi2Constraints}")

        return (pars, cov, chi2, ndf)
