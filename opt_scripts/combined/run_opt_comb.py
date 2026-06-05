import os
import numpy as np
import openmdao.api as om
from funtofem.mphys import MeldBuilder
from openaerostruct.geometry.utils import generate_vsp_surfaces
from openaerostruct.mphys import AeroBuilder
from tacs.mphys import TacsBuilder
from mphys import MPhysVariables, Multipoint
from mphys.scenarios import ScenarioAeroStructural
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import DVGeometryVSP
import sys

DIR        = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIR, "..", "..", "input_files"))
import tacs_setup

# ----------------------- General ---------------------------- #
maxiter = 500

VSP_FILE = os.path.join(DIR, "..", "..", "input_files", "B737SB.vsp3")
BDF_FILE = os.path.join(DIR, "..", "..", "input_files", "B737SB.bdf")
OUTPUT_DIR = os.path.join(DIR, "output")
OUTPUT_AERO = os.path.join(OUTPUT_DIR, "vlm_out")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(OUTPUT_AERO):
    os.makedirs(OUTPUT_AERO, exist_ok=True)

MACH  = 0.78            # Design mach of B737            
RHO   = 0.33            # at 38'000 ft
V     = 235             # m/s
RE    = 1e6
AOA_CRUISE = 3.0        # deg
AOA_MAN    = 8.0        # deg
YAW   = 0.0             # deg

M_GROSS    = 65770.8    # B737-800 gross mass
G          = 9.81       # gravitational
W_GROSS    = M_GROSS*G  # gross weight N
L_CRUISE   = W_GROSS    # Lift constraint
L_MAN      = 2.5*W_GROSS    # Lift constraint
S_REF      = 130        # m^2 (of B737-800)
T_OVER_C   = 0.12

LD_TARGET = 30

CDREF   = 0.06
D_REF   = L_CRUISE/LD_TARGET
M_REF = 1000 # KG
M_REF_UNIT = 50 # kg
D_REF_UNIT = 1000  # N

M_INIT = 3500 # kg
D_INIT = 28000 # N

# ----------------------- Multipoint ---------------------------- #
class Top(Multipoint):

    def setup(self):
        surfaces = generate_vsp_surfaces(VSP_FILE, symmetry=True, include=["Wing", "Strut"])
        for surf in surfaces:
            if surf["name"]=="Wing":
                surf.update({
                    "type": "aero", 
                    "S_ref_type": "wetted",
                    "CL0": 0.0, 
                    "CD0": 0.0,
                    "with_viscous": True, 
                    "with_wave": False,
                    "k_lam": 0.05, 
                    "t_over_c_cp": np.array([T_OVER_C]),
                    "c_max_t": 0.30,
                    "output_dir": OUTPUT_AERO,
                })
            if surf["name"]=="Strut":
                surf.update({
                    "type": "aero", 
                    "S_ref_type": "wetted",
                    "CL0": 0.0, 
                    "CD0": 0.0,
                    "with_viscous": True, 
                    "with_wave": False,
                    "k_lam": 0.05, 
                    "t_over_c_cp": np.array([T_OVER_C]),
                    "c_max_t": 0.30,
                    "output_dir": OUTPUT_AERO,
                })
        
        # Design Variables
        dvgeo_comp = OM_DVGEOCOMP(file=VSP_FILE, type="vsp", options={"comps": ["Wing"]})
        dvgeo_comp.nom_add_discipline_coords(MPhysVariables.Aerodynamics.Surface.Geometry)
        dvgeo_comp.nom_add_discipline_coords(MPhysVariables.Structures.Geometry)
        self.add_subsystem("dvgeo", dvgeo_comp, promotes=["*"])

        # Builders (Aero, Structural, Load/Disp-Transfer)
        aero_builder = AeroBuilder(surfaces, options={"output_dir": OUTPUT_AERO, "write_solution": True})
        aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        struct_builder = TacsBuilder(
            mesh_file=BDF_FILE,
            element_callback=tacs_setup.element_callback,
            problem_setup=tacs_setup.problem_setup,
            constraint_setup=tacs_setup.constraint_setup,
            coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC],
        )
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()
        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=1)
        ldxfer_builder.initialize(self.comm)

        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        # In setup():
        dvs.add_output("aoa_cruise",   val=AOA_CRUISE, units="deg")
        dvs.add_output("aoa_maneuver", val=AOA_MAN,    units="deg")
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE, val=YAW, units="deg")
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER, val=MACH)
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER, val=RE, units="1/m")
        dvs.add_output("rho", val=RHO, units="kg/m**3")
        dvs.add_output("v",   val=V,   units="m/s")
        dvs.add_output("dv_struct", np.full(ndv_struct, 0.002))
        dvs.add_output("Wing:XSec_1:Twist", val=0.0, units="deg")
        dvs.add_output("Wing:XSec_2:Twist", val=0.0, units="deg")

        # Szenario + Solver definieren
        for scenario in ["cruise", "maneuver"]: # , "maneuver"
            # Solver settings (sonst default)
            nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
            linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-3)
            # linear_solver = om.ScipyKrylov(maxiter=25, atol=1e-14, rtol=1e-14, iprint=2)

            self.mphys_add_scenario(
                scenario,
                ScenarioAeroStructural(
                    aero_builder=aero_builder,
                    struct_builder=struct_builder,
                    ldxfer_builder=ldxfer_builder,
                ),
                nonlinear_solver,
                linear_solver,
            )

        # Multiplikative Formulierung
        # self.add_subsystem("obj", om.ExecComp(
        #     "D_M_obj = (wing_mass / M_REF) * (D / D_REF)",
        #     wing_mass=M_REF, D=D_REF,
        #     M_REF=M_REF, D_REF=D_REF
        # ), promotes=["*"])

        # Additive Formulierung
        self.add_subsystem("obj", om.ExecComp(
            "D_M_obj = beta*(wing_mass / M_REF) + ((1-beta) * D / D_REF)",
            wing_mass=M_INIT, D=D_INIT,
            M_REF=M_REF, D_REF=D_REF,
            beta=0.1
        ), promotes=["*"])

        # Breguet-Proxy Objective:
        # self.add_subsystem("obj", om.ExecComp(
        #     "D_M_obj = D + wing_mass * g / LD_ref",
        #     D=D_INIT, wing_mass=M_INIT,
        #     g=G, LD_ref=LD_TARGET
        # ), promotes=["*"])


        # Stall limit AoA
        self.add_subsystem("aoa_total", om.ExecComp(
            "aoa_total_man = aoa_maneuver + twist_inner + twist_outer",
            aoa_maneuver=AOA_MAN, twist_inner =0.0, twist_outer=0.0
        ), promotes=["*"])


# ----------------------- Connect ---------------------------- #

        self.connect(f"mesh_aero.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}",MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,)
        self.connect(f"mesh_struct.{MPhysVariables.Structures.Mesh.COORDINATES}",MPhysVariables.Structures.Geometry.COORDINATES_INPUT,)

        self.connect("aoa_cruise",   f"cruise.{MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK}")
        self.connect("aoa_maneuver", f"maneuver.{MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK}")

        self.connect("cruise.mass", "wing_mass")
        self.connect("cruise.D",    "D")

        self.connect("Wing:XSec_1:Twist", "twist_inner")
        self.connect("Wing:XSec_2:Twist", "twist_outer")

        for scenario in ["cruise", "maneuver"]: #, "maneuver"
            self.connect("x_aero0_geometry_output", f"{scenario}.{MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL}",)
            self.connect("dv_struct", f"{scenario}.dv_struct")
            self.connect("x_struct0_geometry_output", f"{scenario}.{MPhysVariables.Structures.COORDINATES}",)

            for dv in [
                MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE,
                MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER,
                MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER,
                "rho",
                "v",
            ]:
                self.connect(dv, f"{scenario}.{dv}")
            

# ----------------------- DVgeo ---------------------------- #

    def configure(self):
        # for scenario in ["cruise", "maneuver"]:
        #     system = getattr(self, scenario)
        #     system.coupling.linear_solver = om.ScipyKrylov(
        #         maxiter=25, atol=1e-6, restart=50, iprint=2, rhs_checking=True
        #     )
        #     system.coupling.linear_solver.precon = om.LinearBlockGS(maxiter=3)

        self.dvgeo.nom_addVSPVariable("Wing", "XSec_1", "Twist", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_1", "Span", scaledStep=False, dh=1e-3)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_1", "Root_Chord", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_1", "Tip_Chord", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_1", "Sweep", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_2", "Twist", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_2", "Span", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_2", "Tip_Chord", scaledStep=False, dh=1e-3,)
        self.dvgeo.nom_addVSPVariable("Wing", "XSec_2", "Sweep", scaledStep=False, dh=1e-3,)



# ----------------------- Problem setup ---------------------------- #
prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.opt_settings = {"MAXIT": maxiter}
# prob.driver.declare_coloring()

# pyOptSparse history file (for OptView)
prob.driver.options["hist_file"] = os.path.join(OUTPUT_DIR, "opthist.hst")



# Recording Options
prob.driver.recording_options["record_objectives"] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.recording_options["record_desvars"] = True

# SQL-History file
# SQL-History file
sql_file = os.path.join(OUTPUT_DIR, "opt_history.sql")
recorder = om.SqliteRecorder(sql_file)
prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["cruise.mass", "cruise.D"]

# Add Designvariables
ndv_struct = 102
lower, upper = tacs_setup.get_dv_bounds(ndv_struct)
prob.model.add_design_var("dv_struct", lower=lower, upper=upper, scaler=100.0)
prob.model.add_design_var("aoa_cruise",   lower=1.0, upper=5.0,  scaler=1/AOA_CRUISE)
prob.model.add_design_var("aoa_maneuver", lower= 5.0, upper=12.0, scaler=1/AOA_MAN)

prob.model.add_design_var("Wing:XSec_1:Twist", lower=-2.0, upper=5.0, scaler=1/5)
prob.model.add_design_var("Wing:XSec_1:Root_Chord",  lower=6.0, upper=8.0, scaler=1.0/7.7)
# prob.model.add_design_var("Wing:XSec_1:Tip_Chord",  lower=3.0, upper=5.0, scaler=1.0/4)
# prob.model.add_design_var("Wing:XSec_1:Sweep",  lower=0.0, upper=30.0, scaler=1.0/28)
prob.model.add_design_var("Wing:XSec_2:Tip_Chord",  lower=1.0, upper=3.0, scaler=1.0/1.7)
prob.model.add_design_var("Wing:XSec_2:Twist", lower=-2.0, upper=5.0, scaler=1/5)
prob.model.add_design_var("Wing:XSec_2:Span",  lower=5.0, upper=20.0, scaler=1.0/9.5)
prob.model.add_design_var("Wing:XSec_2:Sweep",  lower=0.0, upper=30.0, scaler=1.0/28)

prob.model.add_objective("D_M_obj", scaler=1.0) # 1.0 for addition, 1.0 for multiplication, 1/D_INIT for breguet

# Add constraints
prob.model.add_constraint("aoa_total_man", upper=14.0, scaler=1/14.0) 
prob.model.add_constraint("maneuver.ks_vmfailure", upper=1.0, scaler=1.0)
prob.model.add_constraint("cruise.L", equals=L_CRUISE, scaler=1/L_CRUISE)
prob.model.add_constraint("maneuver.L", equals=L_MAN, scaler=1/L_MAN)
for comp in ["U_SKIN", "L_SKIN", "F_SPAR", "R_SPAR"]:
    prob.model.add_constraint(f"cruise.adjacency.{comp}", lower=-2.5e-3, upper=2.5e-3, scaler=1e3, linear=True)
# prob.model.add_constraint("cruise.Wing.S_ref", lower=60, upper=220.0, scaler=1/S_REF)

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile=os.path.join(OUTPUT_DIR, "n2.html"))

assert len(prob.get_val("dv_struct")) == ndv_struct, \
    f"ndv_struct stimmt nicht! – Indizes in get_dv_bounds prüfen!"

for scenario in ["cruise", "maneuver"]: #, "maneuver"
    system = getattr(prob.model, scenario)
    system.coupling.linear_solver = om.LinearBlockGS(maxiter=6, atol=1e-6, rtol=1e-3, iprint=2)
    # system.coupling.linear_solver = om.ScipyKrylov(maxiter=25, atol=1e-6, restart=50, iprint=2, rhs_checking=True)
    # system.coupling.linear_solver.precon = om.LinearBlockGS(maxiter=5)

for scenario in ["cruise", "maneuver"]:
    for surf in ["Wing", "Strut"]:
        varname = f"{scenario}.aero_post.{surf}.viscousdrag.t_over_c"
        current = prob.get_val(varname)
        prob.set_val(varname, np.full(current.shape, T_OVER_C))

prob.run_driver()
# prob.run_model()
prob.cleanup()

# data = prob.check_partials(
#     compact_print=True,
#     includes=["dvgeo"],
#     out_stream=None  # kein Print
# )

# # Nur Fehler manuell ausgeben
# for comp, comp_data in data.items():
#     for (of, wrt), deriv in comp_data.items():
#         rel_err = deriv["rel error"].forward
#         if rel_err > 1e-4:
#             print(f"{comp}: d({of})/d({wrt}) rel_err={rel_err:.2e}")

# prob.check_partials(compact_print=True, show_only_incorrect=True)
prob.check_totals(
    of=["maneuver.ks_vmfailure"],
    wrt=["dv_struct"],
    compact_print=True
)
# prob.check_totals(compact_print=True, show_only_incorrect=True)

# Property-Namen auf Designvariablen Patchen
dv_values = prob.get_val("dv_struct")
fea_assembler = prob.model.cruise.struct_pre.distributor.fea_assembler
dv_names = fea_assembler.getCompNames()

print("\n" + "=" * 55)
print("DESIGNVARIABLEN – TACS Property Mapping")
print("=" * 55)
for i, (name, val) in enumerate(zip(dv_names, dv_values)):
    print(f"  [{i:3d}] {name:<35} {val*1000:6.2f} mm")
print("=" * 55)

# Output-geometrien schreiben
dvgeo_internal = prob.model.dvgeo.nom_getDVGeo()
dvgeo_internal.writeVSPFile(os.path.join(OUTPUT_DIR, "comb_out.vsp3"))

# Results in .txt file festhalten
output_file = os.path.join(OUTPUT_DIR, "results.txt")
with open(output_file, "w") as f:
    old_stdout = sys.stdout
    sys.stdout = f

    # Vollständige Modell-Outputs (wie gehabt)
    prob.model.list_outputs()

    # --- ALLE Design-Variablen zwingend auflisten ---
    print("\n" + "=" * 60)
    print("ALLE DESIGN-VARIABLEN (vom Treiber registriert)")
    print("=" * 60)
    dv_meta = prob.model.get_design_vars(recurse=True)
    for name in sorted(dv_meta):
        val = prob.get_val(name)
        arr = np.atleast_1d(val)
        if arr.size == 1:
            print(f"  {name:<35} {arr[0]: .6f}")
        else:
            print(f"  {name:<35} size={arr.size:<4}  "
                  f"min={arr.min(): .4f}  max={arr.max(): .4f}  mean={arr.mean(): .4f}")
    print("=" * 60)

    sys.stdout = old_stdout

    # Strukturdicken mit Namen (wie gehabt)
    f.write("\n" + "=" * 55 + "\n")
    f.write("STRUKTURDICKEN – TACS Property Mapping\n")
    f.write("=" * 55 + "\n")
    dv_values = prob.get_val("dv_struct")
    dv_names = fea_assembler.getCompNames()
    for i, (name, val) in enumerate(zip(dv_names, dv_values)):
        f.write(f"  [{i:3d}] {name:<35} {val*1000:6.2f} mm\n")

inner_span = prob.get_val("Wing:XSec_1:Span")[0]
outer_span = prob.get_val("Wing:XSec_2:Span")[0]
Total_L = prob.get_val('cruise.L')[0]
Total_D = prob.get_val('cruise.D')[0]

print("\n" + "=" * 65)
print("OPTIMIERUNGSERGEBNIS  –  B737SB Cruise, Maneuver Mass-Drag-Combined ")
print("=" * 65)
print(f"  Strukturmasse              : {prob.get_val('cruise.mass')[0]:>10.2f}  kg")
print(f"  Auftriebskraft             : {prob.get_val('cruise.L')[0]:>10.2f}  N")
print(f"  S_ref                      : {prob.get_val('cruise.Wing.S_ref')[0]:>10.2f}  m^2")
print(f"  Total Drag                 : {prob.get_val('cruise.D')[0]:>10.2f}  N")
print(f"  CL                         : {prob.get_val('cruise.CL')[0]:>10.4f}  ")
print(f"  CD                         : {prob.get_val('cruise.CD')[0]:>10.4f}  ")
print(f"  Wing CD visc               : {prob.get_val('cruise.aero_post.Wing.CDv')[0]:>10.4f}  ")
print(f"  Wing CD ind                : {prob.get_val('cruise.aero_post.Wing.CDi')[0]:>10.4f}  ")
print(f"  Strut CD visc              : {prob.get_val('cruise.aero_post.Strut.CDv')[0]:>10.4f}  ")
print(f"  Strut CD ind               : {prob.get_val('cruise.aero_post.Strut.CDi')[0]:>10.4f}  ")
print(f"  Total L/D                  : {Total_L/Total_D:>10.3f}")
print(f"  Wing Lift                  : {prob.get_val('cruise.aero_post.Wing.L')}")
print(f"  Strut Lift                 : {prob.get_val('cruise.aero_post.Strut.L')}")
print(f"  Cruise Lift                : {prob.get_val('cruise.L')}")
print(f"  KS-Failure                 : {prob.get_val('cruise.ks_vmfailure')[0]:>10.4f}  (≤ 1.0)")
print(f"  AoA Cruise                 : {prob.get_val('aoa_cruise')[0]:>10.3f}  deg")
print(f"  AoA Maneuver               : {prob.get_val('aoa_maneuver')[0]:>10.3f}  deg")
print(f"  Wing Twist inner           : {prob.get_val('Wing:XSec_1:Twist')[0]:>10.3f}  deg")
print(f"  Wing Twist outer           : {prob.get_val('Wing:XSec_2:Twist')[0]:>10.3f}  deg")
print(f"  Wing Span (Halb)           : {inner_span+outer_span:>10.3f}  m")
print(f"  Wing Chord (Root)          : {prob.get_val('Wing:XSec_1:Root_Chord')[0]:>10.3f}  m")
print(f"  Wing Chord (Tip)           : {prob.get_val('Wing:XSec_2:Tip_Chord')[0]:>10.3f}  m")
dv = prob.get_val("dv_struct")
print(f"  Dicken  min / max / mean   :  {dv.min():.4f} / {dv.max():.4f} / {dv.mean():.4f}  m")
print("=" * 65)
print(f"  N2-Diagram  : {os.path.join(OUTPUT_DIR, 'n2.html')}")

# prob.model.list_outputs()