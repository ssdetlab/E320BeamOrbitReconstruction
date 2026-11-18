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
    # SM scan, no steering
    bpm0ResolutionEstLong = 4.91 * 1e-6
    bpm1ResolutionEstLong = 5.89 * 1e-6
    bpm2ResolutionEstLong = 14.12 * 1e-6
    bpm3ResolutionEstLong = 13.86 * 1e-6

    bpm0ResolutionEstShort = 4.91 * 1e-6
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

    # PICENT location
    piccentCenterPrimary = 1992.825

    # Al foil location
    alFoilCenterPrimary = 1992.325

    # Beam at the reference within one bunch 
    # (0 -> no contribution to the measurement covariance)
    # size [m], divergence [rad], size-divergence correlation coefficient
    referenceBeamSizeLong = 0.0
    referenceBeamDivergenceLong = 0.0
    referenceBeamCorrelationLong = 0.0

    referenceBeamSizeShort = 0.0
    referenceBeamDivergenceShort = 0.0
    referenceBeamCorrelationShort = 0.0
