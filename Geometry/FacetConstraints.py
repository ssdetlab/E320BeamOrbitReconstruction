class FacetConstraints:
    # Quads dimensions
    quad1HalfPrimary = 486.664 * 1e-3
    quad2HalfPrimary = 486.664 * 1e-3
    quad3HalfPrimary = 486.664 * 1e-3

    quad1HalfLong = 2000 * 1e-3
    quad2HalfLong = 2000 * 1e-3
    quad3HalfLong = 2000 * 1e-3

    quad1HalfShort = 2000 * 1e-3
    quad2HalfShort = 2000 * 1e-3
    quad3HalfShort = 2000 * 1e-3

    # --------------------------------------------------------------
    # Doug's survey parameters
    
    # Parameters of the BPMs
    bpm0CenterPrimary = 1991.310
    bpm1CenterPrimary = 1997.769
    bpm2CenterPrimary = 1999.993
    bpm3CenterPrimary = 2002.218
    
    bpm0CenterLong = 0.0
    bpm1CenterLong = 0.0
    bpm2CenterLong = 0.0
    bpm3CenterLong = 0.0
    
    bpm0CenterShort = 0.0
    bpm1CenterShort = 0.0
    bpm2CenterShort = 0.0
    bpm3CenterShort = 0.0

    # Survey resolution of the BPMs [m]
    bpm0ResolutionSurveyLong = 0.05 * 1e-3
    bpm1ResolutionSurveyLong = 0.05 * 1e-3
    bpm2ResolutionSurveyLong = 0.15 * 1e-3
    bpm3ResolutionSurveyLong = 0.15 * 1e-3

    bpm0ResolutionSurveyShort = 0.05 * 1e-3
    bpm1ResolutionSurveyShort = 0.05 * 1e-3
    bpm2ResolutionSurceyShort = 0.15 * 1e-3
    bpm3ResolutionSurveyShort = 0.15 * 1e-3

    # Estimated resolution of the BPMs [m]
    # # SM scan, no steering
    # bpm0ResolutionEstLong = 4.91 * 1e-6
    # bpm1ResolutionEstLong = 5.89 * 1e-6
    # bpm2ResolutionEstLong = 14.12 * 1e-6
    # bpm3ResolutionEstLong = 13.86 * 1e-6

    # bpm0ResolutionEstShort = 4.91 * 1e-6
    # bpm1ResolutionEstShort = 6.45 * 1e-6
    # bpm2ResolutionEstShort = 15.06 * 1e-6
    # bpm3ResolutionEstShort = 14.02 * 1e-6

    # SM scan, no steering, SM steering steering response
    bpm0ResolutionEstLong = 5.63 * 1e-6
    bpm1ResolutionEstLong = 6.07 * 1e-6
    bpm2ResolutionEstLong = 14.11 * 1e-6
    bpm3ResolutionEstLong = 13.80 * 1e-6

    bpm0ResolutionEstShort = 5.63 * 1e-6
    bpm1ResolutionEstShort = 6.45 * 1e-6
    bpm2ResolutionEstShort = 15.06 * 1e-6
    bpm3ResolutionEstShort = 14.02 * 1e-6

    # Parameters of the quads
    quad0CenterPrimary = 1990.925
    quad1CenterPrimary = 1996.982
    quad2CenterPrimary = 1999.207
    quad3CenterPrimary = 2001.431
	
    quad0CenterLong = 0.0
    quad1CenterLong = -0.194 * 1e-3
    quad2CenterLong = -0.253 * 1e-3
    quad3CenterLong = -0.146 * 1e-3

    quad0CenterShort = 0.0
    quad1CenterShort = 3.429 * 1e-3
    quad2CenterShort = 2.633 * 1e-3
    quad3CenterShort = 3.601 * 1e-3

    # Rotation [rad]
    quad1Pitch = 0
    quad2Pitch = 0
    quad3Pitch = 0

    quad1Yaw = 0
    quad2Yaw = 0
    quad3Yaw = 0

    # Survey precision of the quads, used as the widths of the 
    # survey constraints in the alignment fits
    quadSurveySigmaShift = 100 * 1e-6 # [m]
    quadSurveySigmaTilt = 1 * 1e-3 # [rad]

    # PICENT location
    piccentCenterPrimary = 1992.825

    # Al foil location
    alFoilCenterPrimary = 1992.325

    # Steering response J: change of the effective alignment parameters 
    # per unit change of the beam vector (x, x') at BPM0, rows in the 
    # getAlignmentPars order, columns (x, x') [m/m, m/rad; rad/m, rad/rad] 
    # (setSteeringResponse, 0 -> no correction; 
    # estimate with SteeringResponseEstimationRunCsv.py)
    steeringResponseLong = [
        [0.000000e+00, 0.0], # bpm0 shift
        [0.000000e+00, 0.0], # bpm1 shift
        [-2.670746e-01, 0.0], # bpm2 shift
        [-2.916478e-01, 0.0], # bpm3 shift
        [-1.678677e-01, 0.0], # quad1 shift
        [-3.535279e-01, 0.0], # quad2 shift
        [-2.407361e-01, 0.0], # quad3 shift
        [0.000000e+00, 0.0], # quad1 tilt
        [0.000000e+00, 0.0], # quad2 tilt
        [0.000000e+00, 0.0], # quad3 tilt
    ]
    steeringResponseShort = [
        [0.0, 0.0], # bpm0 shift
        [0.0, 0.0], # bpm1 shift
        [0.0, 0.0], # bpm2 shift
        [0.0, 0.0], # bpm3 shift
        [0.0, 0.0], # quad1 shift
        [0.0, 0.0], # quad2 shift
        [0.0, 0.0], # quad3 shift
        [0.0, 0.0], # quad1 tilt
        [0.0, 0.0], # quad2 tilt
        [0.0, 0.0], # quad3 tilt
    ]

    # Gradient dependent systematic of the reference orbit: (x, x') covariance 
    # at the reference [m^2, m rad; m rad, rad^2] (setGradientSystematicCov, 
    # 0 -> no contribution; estimate with GradientSystematicEstimationRunCsv.py)
    gradientSystematicCovLong = [
        [2.667868e-11, -1.208210e-11],
        [-1.208210e-11, 1.530671e-11],
    ]
    gradientSystematicCovShort = [
        [6.307852e-11, -3.136790e-11],
        [-3.136790e-11, 3.048678e-11],
    ]

    # Beam bunch parameters at the reference: spot size [m], 
    # angular divergence [rad], size-divergence correlation coefficient
    # (setBeamBunchCov, 0 -> zero bunch covariance)
    referenceBeamSizeLong = 0.03 * 1e-3
    referenceBeamDivergenceLong = 1.0 * 1e-6
    referenceBeamCorrelationLong = 0.0

    referenceBeamSizeShort = 0.03 * 1e-3
    referenceBeamDivergenceShort = 1.0 * 1e-6
    referenceBeamCorrelationShort = 0.0

    # Alignment samples: the orbit covariance at the reference is 
    # diag(RMS^2) of the sample shifts about the current alignment 
    # for more than alignmentSampleRmsThreshold samples, 
    # diag(half-range^2) otherwise (setAlignmentSampleRmsThreshold)
    alignmentSampleRmsThreshold = 10
