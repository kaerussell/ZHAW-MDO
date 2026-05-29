import numpy as np
from tacs import constitutive, elements, functions
import os

# Output Directory to be created or found:
output_dir = "output/f5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Gravitational accel.
g_scalar = 9.81     # m/s^2
g = np.array([0.0, 0.0, -g_scalar])  
load_factor = 2.5

# fuel
m_fuel = 6000  # kg
rho_fuel = 800  # kg/m^3
h_fuel = 0.4    # m
fill_maneuver = 0.3 # % fuel during maneuver

# engine
m_engine = 2400 # kg, B737-800 engine (CFM 56-7B)
l_engine = 1    # m, engine lever arm

# Material properties
rho = 2780.0  # density kg/m^3
E = 70.0e9  # Young's modulus (Pa)
nu = 0.30  # Poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress

# Shell thickness
t = 0.0025  # m

# Iso Tube Beam
tTube = 0.0025 # m
d = 0.15 # m

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)

    # Component specific Bounds
    if "SKIN" in compDescript:
        tMin, tMax = 0.001, 0.010  # 1-10mm
    elif "SPAR" in compDescript:
        tMin, tMax = 0.002, 0.050  # 2-50mm
    elif "RIB" in compDescript:
        tMin, tMax = 0.001, 0.025  # 1-25mm
    elif "STRUT" in compDescript:
        tMin, tMax = 0.001, 0.025  # 1-25mm
    else:
        tMin, tMax = 0.001, 0.050

    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum, tMin=tMin, tMax=tMax)
    conBeam = constitutive.IsoTubeBeamConstitutive(prop,t=tTube, tMax=tMax, tNum=dvNum, d=d, dNum=-1)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    transform = None

    for elemDescript in elemDescripts:
        if elemDescript in ["CQUAD4", "CQUADR"]:
            elem = elements.Quad4Shell(transform, con)
        elif elemDescript in ["CTRIA3", "CTRIAR"]:
            elem = elements.Tri3Shell(transform, con)
        elif elemDescript in ["CROD", "CBEAM"]:
            refAxis = np.array([0.0, 0.0, 1.0])
            transformBeam = elements.BeamRefAxisTransform(refAxis)
            elem = elements.Beam2(transformBeam, conBeam)
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        elemList.append(elem)

    # Add scale for thickness dv
    scale = [100.0]
    return elemList, scale


def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Option for Output Directory
    problem.setOption("outputDir", output_dir)
    
    # Add TACS Functions
    problem.addFunction("mass", functions.StructuralMass)
    problem.addFunction(
        "ks_vmfailure", functions.KSFailure, safetyFactor=1.2, ksWeight=50.0
    )

    # engine load
    F_engine = m_engine * g_scalar
    compIDs_engine = fea_assembler.selectCompIDs(include="RIB/SEG.05")
    problem.addLoadToComponents(
        compIDs_engine,
        F=[0, 0, -F_engine, 0, -F_engine * l_engine, 0]
    )

    # fuel load
    compIDs_fuel = []
    for i in range(1, 8):
        compID = fea_assembler.selectCompIDs(include=f"L_SKIN/SEG.0{i}")
        compIDs_fuel.extend(compID)
    P = rho_fuel * g_scalar * h_fuel  # N/m²
    problem.addPressureToComponents(compIDs_fuel, P)

    if scenario_name == "cruise":
        F_fuel = m_fuel * g_scalar # N
        problem.addLoadToComponents(compIDs_fuel,F=[0, 0, -F_fuel, 0, 0, 0],averageLoad=True)

    # Inertial Loads
    if scenario_name == "maneuver":
        F_fuel = m_fuel * load_factor * g_scalar # N
        problem.addLoadToComponents(compIDs_fuel,F=[0, 0, -F_fuel*fill_maneuver, 0, 0, 0],averageLoad=True)
        problem.addInertialLoad(load_factor * g)

def constraint_setup(scenario_name, fea_assembler, constraint_list):
    """
    Helper function to setup tacs constraint classes
    """
    if scenario_name in ("maneuver", "cruise"):
        # Setup adjacency constraints for skin and spar panel thicknesses
        constr = fea_assembler.createAdjacencyConstraint("adjacency")
        compIDs = fea_assembler.selectCompIDs(include="U_SKIN")
        constr.addConstraint("U_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="L_SKIN")
        constr.addConstraint("L_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="F_SPAR")
        constr.addConstraint("F_SPAR", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="R_SPAR")
        constr.addConstraint("R_SPAR", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="STRUT")
        constr.addConstraint("STRUT", compIDs=compIDs)
        constraint_list.append(constr)