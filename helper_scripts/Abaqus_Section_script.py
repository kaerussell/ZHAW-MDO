# wingbox_sections_per_face_v4.py
# One section per face, sorted by Y (1=root, N=tip)
# Section names follow pyTACS convention: GROUP/SUBGROUP/SEG.XX
# Abaqus Python 2.7 - no print output - no group sets
import abaqusConstants as C
from abaqus import mdb

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
MODEL_NAME    = 'Model-1'
PART_NAME     = 'wing4_wingbox'

T_SKIN = 0.003
T_RIB  = 0.003
T_SPAR = 0.003

# (set_name, tacs_group, tacs_subgroup, thickness)
# Section name will be: tacs_group/tacs_subgroup/SEG.XX
# e.g. WING_SKINS/U_SKIN/SEG.01, WING_SPARS/F_SPAR/SEG.01
COMPONENT_SETS = [
    ('U_Skin', 'SKINS', 'U_SKIN', T_SKIN),
    ('L_Skin', 'SKINS', 'L_SKIN', T_SKIN),
    ('Ribs',   'RIBS',  'RIB',    T_RIB),
    ('F_Spar', 'SPARS', 'F_SPAR', T_SPAR),
    ('R_Spar', 'SPARS', 'R_SPAR', T_SPAR),
]

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
model = mdb.models[MODEL_NAME]
part  = model.parts[PART_NAME]

total = 0

for (set_name, tacs_group, tacs_subgroup, thickness) in COMPONENT_SETS:
    if set_name not in part.sets.keys():
        continue

    faces = part.sets[set_name].faces
    n     = len(faces)

    # Sort faces by Y-coordinate ascending (1 = root, N = tip)
    face_list = []
    for i in range(n):
        pt = faces[i].pointOn[0]
        face_list.append((pt[1], pt))
    face_list.sort(key=lambda x: x[0])

    for rank in range(len(face_list)):
        pt       = face_list[rank][1]
        seg_num  = str(rank + 1).zfill(2)

        # pyTACS-compatible section name: GROUP/SUBGROUP/SEG.XX
        sec_name      = tacs_group + '/' + tacs_subgroup + '/SEG.' + seg_num
        face_set_name = tacs_group + "-" + tacs_subgroup + "-SEG_" + seg_num

        # Single-face set
        face_obj = part.faces.findAt((pt,))
        if face_set_name in part.sets.keys():
            del part.sets[face_set_name]
        face_set = part.Set(name=face_set_name, faces=face_obj)

        # Shell section
        if sec_name in model.sections.keys():
            del model.sections[sec_name]
        model.HomogeneousShellSection(
            name                = sec_name,
            material            = '',
            thickness           = thickness,
            thicknessType       = C.UNIFORM,
            preIntegrate        = C.OFF,
            poissonDefinition   = C.DEFAULT,
            nodalThicknessField = '',
            thicknessField      = '',
            integrationRule     = C.SIMPSON,
            numIntPts           = 5,
            temperature         = C.GRADIENT,
            useDensity          = C.OFF,
        )

        # Section assignment
        part.SectionAssignment(
            region              = face_set,
            sectionName         = sec_name,
            offset              = 0.0,
            offsetType          = C.MIDDLE_SURFACE,

            thicknessAssignment = C.FROM_SECTION,
        )

        total += 1

print('Done: ' + str(total))