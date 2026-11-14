import csv
from Io.FacetDataset import FacetDataset

import ROOT
import numpy as np
from Beamline import *

ROOT.gSystem.Load("/home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so")

# ------------------------------------------------------
# CSV output
outFile = open("outFile.csv", "w", newline="")
writer = csv.writer(outFile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

# ------------------------------------------------------
# FACET-II files
filePaths = [
        "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_14066/BEAMPHYS_14066.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_14067/BEAMPHYS_14067.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_14068/BEAMPHYS_14068.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_18565/BEAMPHYS_18565.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_18651/BEAMPHYS_18651.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/BEAMPHYS_18652/BEAMPHYS_18652.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17564/E320_17564.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17659/E320_17659.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17774/E320_17774.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17962/E320_17962.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17967/E320_17967.mat",
        # "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_bpm_analysis/quad_scan_data_facet/E320_17995/E320_17995.mat",
        ]

for filePath in filePaths:
    data = FacetDataset(filePath)
        
    scalars = data.data_struct.scalars
    indices = data.data_struct.scalars.common_index
    # indicesSCP = data.data_struct.scalars.common_index_SCP
    # indicesInclSCP = data.data_struct.scalars.common_index_inclSCP
    steps = data.data_struct.scalars.steps

    print("-"*50)
    print(filePath)

    print("-"*50)
    for key in scalars.value.keys():
        print(key)
    print("-"*50)
    
    print("-"*50)
    for key in scalars.value["BSA_List_S20"].keys():
        print(key)
    print("-"*50)
    
    print("-"*50)
    for key in scalars.value["nonBSA_List_S20Magnets"].keys():
        print(key)
    print("-"*50)
    
    bsa = scalars.value["BSA_List_S20"]
    magnets = scalars.value["nonBSA_List_S20Magnets"]
    

    steerXcor = magnets["LI20_XCOR_3116_BACT"][indices]
    print(magnets["LI20_XCOR_3116_BACT"][indices])
    print(magnets["LI20_YCOR_3147_BACT"][indices])

    bpm3156_X = bsa["BPMS_LI20_3156_X"][indices] * 1e-3
    bpm3218_X = bsa["BPMS_LI20_3218_X"][indices] * 1e-3
    bpm3265_X = bsa["BPMS_LI20_3265_X"][indices] * 1e-3
    bpm3315_X = bsa["BPMS_LI20_3315_X"][indices] * 1e-3

    bpm3156_Y = bsa["BPMS_LI20_3156_Y"][indices] * 1e-3
    bpm3218_Y = bsa["BPMS_LI20_3218_Y"][indices] * 1e-3
    bpm3265_Y = bsa["BPMS_LI20_3265_Y"][indices] * 1e-3
    bpm3315_Y = bsa["BPMS_LI20_3315_Y"][indices] * 1e-3
    
    quad1 = np.abs(magnets["LI20_LGPS_3141_BACT"][indices]) * 1e-1
    quad2 = np.abs(magnets["LI20_LGPS_3261_BACT"][indices]) * 1e-1
    quad3 = np.abs(magnets["LI20_LGPS_3091_BACT"][indices]) * 1e-1
    print(steps)
    print(quad1)
    print(quad2)
    print(quad3)

    quad1_test = np.abs(magnets["LI20_LGPS_3204_BACT"][indices]) * 1e-1
    quad2_test = np.abs(magnets["LI20_LGPS_3261_BACT"][indices]) * 1e-1
    quad3_test = np.abs(magnets["LI20_LGPS_3311_BACT"][indices]) * 1e-1
    print(quad1_test)
    print(quad2_test)
    print(quad3_test)

    for i in range(len(indices)):
        writer.writerow([
            bpm3156_X[i], 
            bpm3218_X[i], 
            bpm3265_X[i], 
            bpm3315_X[i], 
            bpm3156_Y[i], 
            bpm3218_Y[i], 
            bpm3265_Y[i], 
            bpm3315_Y[i], 
            quad1[i], 
            quad2[i], 
            quad3[i], 
            steerXcor[i]])
    
    print(len(bpm3156_X))
    print(len(bpm3218_X))
    print(len(bpm3265_X))
    print(len(bpm3315_X))
