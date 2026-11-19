from Beamline.E320Beamline import E320Beamline
from Beamline.FreePropagation import FreePropagation
from Beamline.Quadrupole import FocusQuadrupole, DefocusQuadrupole
from Types.BeamState import BeamState

import numpy as np

class E320BeamlineFDF(E320Beamline):
    def __init__(self):
        super().__init__() 

    def getBpms(self, start, quadGrads) -> list[float]:
        # Effective alignment for this beam vector (see setSteeringResponse)
        pars = self.getSteeredAlignmentPars(start)
        bpmShift = pars[0:4]
        quadShift = pars[4:7]
        quadTilt = pars[7:10]

        iniVector = np.zeros(shape=(1,2),dtype=float) 
        iniVector[0,0] = start[0]
        iniVector[0,1] = start[1]

        BPM0 = iniVector[0,0] - bpmShift[0]

        quad1Shift = quadShift[0]
        quad1Tilt = quadTilt[0]
        espec = FreePropagation(self.LQ[0])
        espec = FocusQuadrupole(
                quadGrads[0], 
                self.quadLength[0], 
                self.energy, 
                quad1Shift,
                quad1Tilt) * espec
        espec = FreePropagation(self.LB[0]) * espec
        result1 = espec.propagate(iniVector)
        BPM1 = result1[0,0] - bpmShift[1]

        quad2Shift = quadShift[1]
        quad2Tilt = quadTilt[1]
        espec = FreePropagation(self.LQ[1])
        espec = DefocusQuadrupole(
                quadGrads[1], 
                self.quadLength[1],
                self.energy, 
                quad2Shift,
                quad2Tilt) * espec
        espec = FreePropagation(self.LB[1]) * espec
        result2 = espec.propagate(result1)
        BPM2 = result2[0,0] - bpmShift[2]

        quad3Shift = quadShift[2]
        quad3Tilt = quadTilt[2]
        espec = FreePropagation(self.LQ[2]) 
        espec = FocusQuadrupole(
                quadGrads[2], 
                self.quadLength[2], 
                self.energy, 
                quad3Shift,
                quad3Tilt) * espec
        espec = FreePropagation(self.LB[2]) * espec
        result3 = espec.propagate(result2)
        BPM3 = result3[0,0] - bpmShift[3]

        return [BPM0, BPM1, BPM2, BPM3]
    
    def getBeamStates(self, bpmMeas, bpmErrs, vec, cov, quadGrads) -> list[BeamState]:
        """
        Propagate the beam vector and its covariance from BPM0 to the BPMs

        vec: beam vector (x, x') at BPM0 [m, rad]
        cov: covariance of the beam vector at BPM0
        quadGrads: list of quadrupole gradients [T/m]

        return: list of the beam states at the BPMs
        """
        # Effective alignment for this beam vector (see setSteeringResponse)
        pars = self.getSteeredAlignmentPars(vec)
        bpmShift = pars[0:4]
        quadShift = pars[4:7]
        quadTilt = pars[7:10]

        iniVector = np.zeros(shape=(1,2),dtype=float) 
        iniVector[0,0] = vec[0]
        iniVector[0,1] = vec[1]

        states = []
        states.append(BeamState(
                bpmIdx = 0, 
                vec = iniVector[0,:], 
                cov = np.array(cov), 
                bpmShift = bpmShift[0], 
                bpmMeas = bpmMeas[0], 
                bpmErr = bpmErrs[0], 
                energy = self.energy))

        quad1Shift = quadShift[0]
        quad1Tilt = quadTilt[0]
        espec = FreePropagation(self.LQ[0])
        espec = FocusQuadrupole(
                quadGrads[0], 
                self.quadLength[0], 
                self.energy, 
                quad1Shift,
                quad1Tilt) * espec
        espec = FreePropagation(self.LB[0]) * espec
        result1 = espec.propagate(iniVector)
        cov1 = espec.ABCD @ cov @ espec.ABCD.T
        states.append(BeamState(
                bpmIdx = 1, 
                vec = result1[0,:], 
                cov = cov1, 
                bpmShift = bpmShift[1], 
                bpmMeas = bpmMeas[1], 
                bpmErr = bpmErrs[1], 
                energy = self.energy))

        quad2Shift = quadShift[1]
        quad2Tilt = quadTilt[1]
        espec = FreePropagation(self.LQ[1])
        espec = DefocusQuadrupole(
                quadGrads[1], 
                self.quadLength[1],
                self.energy, 
                quad2Shift,
                quad2Tilt) * espec
        espec = FreePropagation(self.LB[1]) * espec
        result2 = espec.propagate(result1)
        cov2 = espec.ABCD @ cov1 @ espec.ABCD.T
        states.append(BeamState(
                bpmIdx = 2, 
                vec = result2[0,:], 
                cov = cov2, 
                bpmShift = bpmShift[2], 
                bpmMeas = bpmMeas[2], 
                bpmErr = bpmErrs[2], 
                energy = self.energy))

        quad3Shift = quadShift[2]
        quad3Tilt = quadTilt[2]
        espec = FreePropagation(self.LQ[2]) 
        espec = FocusQuadrupole(
                quadGrads[2], 
                self.quadLength[2], 
                self.energy, 
                quad3Shift,
                quad3Tilt) * espec
        espec = FreePropagation(self.LB[2]) * espec
        result3 = espec.propagate(result2)
        cov3 = espec.ABCD @ cov2 @ espec.ABCD.T
        states.append(BeamState(
                bpmIdx = 3, 
                vec = result3[0,:], 
                cov = cov3, 
                bpmShift = bpmShift[3], 
                bpmMeas = bpmMeas[3], 
                bpmErr = bpmErrs[3], 
                energy = self.energy))

        return states
