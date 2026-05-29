"""
bdf_rewrite.py
----------------------
Konvertiert eine Abaqus-generierte BDF in ein pyTACS-kompatibles Format.

Input (Abaqus):
    $ PSHELL created from shell section
    $        elset = WING_WING_SPARS-F_SPAR-SEG_10
    $     material = ALU_1
    PSHELL*               46               0 3.000000000e-03               0
    *                                      0

Output:
    $ Shell element data for family  WING_SPARS/F_SPAR/SEG.10
    PSHELL*               46               0 3.000000000e-03               0
    *                                      0

Namens-Transformation (elset -> pyTACS):
    WING_WING_SPARS-F_SPAR-SEG_10  ->  WING_SPARS/F_SPAR/SEG.10
    1. Instance-Prefix entfernen (INSTANCE_PREFIX)
    2. Ersten "-" -> "/"
    3. Zweiten "-" -> "/"
    4. SEG_XX -> SEG.XX

Verwendung:
    python bdf_abaqus_to_tacs.py input.bdf output.bdf
"""

import re
import sys

# ── Konfiguration ─────────────────────────────────────────────────────────────

# Instance-Prefix entfernen, z.B. "WING_" wenn Sets heissen:
# WING_WING_SPARS-F_SPAR-SEG_09 -> WING_SPARS-F_SPAR-SEG_09
# Leer lassen ("") wenn kein Prefix vorhanden
INSTANCE_PREFIX = 'WING_'

# ── Regex ─────────────────────────────────────────────────────────────────────

RE_PSHELL_CREATED = re.compile(r'^\$\s+PSHELL\s+created', re.IGNORECASE)
RE_ELSET          = re.compile(r'^\$\s+elset\s*=\s*(\S+)', re.IGNORECASE)
RE_MATERIAL       = re.compile(r'^\$\s+material\s*=', re.IGNORECASE)
RE_DOLLAR_EMPTY   = re.compile(r'^\$\s*$')
RE_PSHELL         = re.compile(r'^PSHELL\*?\s', re.IGNORECASE)

# ── Namens-Transformation ─────────────────────────────────────────────────────

def transform_name(raw):
    """
    WING_WING_SPARS-F_SPAR-SEG_10  ->  WING_SPARS/F_SPAR/SEG.10
    """
    name = raw
    if INSTANCE_PREFIX and name.startswith(INSTANCE_PREFIX):
        name = name[len(INSTANCE_PREFIX):]
    name = name.replace('-', '/', 2)
    name = re.sub(r'SEG_(\d+)', r'SEG.\1', name, flags=re.IGNORECASE)
    return name

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def is_abaqus_noise(line):
    """Erkennt Abaqus-Kommentare die geloescht werden sollen."""
    if not line.startswith('$'):
        return False
    if RE_PSHELL_CREATED.match(line):
        return True
    if RE_ELSET.match(line):
        return True
    if RE_MATERIAL.match(line):
        return True
    if RE_DOLLAR_EMPTY.match(line):
        return True
    return False

def get_pid(line):
    parts = line.split()
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None

def build_pid_name_mapping(lines):
    """
    Liest $elset-Kommentare vor PSHELL*-Eintraegen und gibt {PID: name} zurueck.
    """
    mapping  = {}
    in_block = False
    pending  = None

    for line in lines:
        if RE_PSHELL_CREATED.match(line):
            in_block = True
            pending  = None
            continue

        if in_block:
            m = RE_ELSET.match(line)
            if m:
                pending = transform_name(m.group(1))
                continue
            if RE_MATERIAL.match(line) or RE_DOLLAR_EMPTY.match(line):
                continue
            if RE_PSHELL.match(line):
                pid = get_pid(line)
                if pid is not None and pending is not None:
                    mapping[pid] = pending
                in_block = False
                pending  = None
                continue
            in_block = False
            pending  = None

    return mapping

# ── Haupt-Logik ───────────────────────────────────────────────────────────────

def process_bdf(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Schritt 1: PID -> Name Mapping aufbauen
    mapping = build_pid_name_mapping(lines)

    if not mapping:
        print('WARNUNG: Kein PID-Mapping gefunden!')
        print('         Pruefe ob $elset-Kommentare vor PSHELL stehen.')
    else:
        print(str(len(mapping)) + ' Properties gefunden:')
        for pid, name in sorted(mapping.items()):
            print('  PID {:>5}  ->  {}'.format(pid, name))

    out_lines       = []
    in_pshell_block = False
    pending_name    = None

    skip_continuation = False

    for line in lines:
        if skip_continuation:
            if line.startswith('*'):
                continue          
        skip_continuation = False  
        # ── Abaqus-Block erkennen ────────────────────────────────────────────
        if RE_PSHELL_CREATED.match(line):
            in_pshell_block = True
            pending_name    = None
            continue

        if in_pshell_block:
            m = RE_ELSET.match(line)
            if m:
                pending_name = transform_name(m.group(1))
                continue
            if RE_MATERIAL.match(line) or RE_DOLLAR_EMPTY.match(line):
                continue
            if RE_PSHELL.match(line):
                pid = get_pid(line)
                name = pending_name or mapping.get(pid)
                if name:
                    out_lines.append('$       Shell element data for family    {}\n'.format(name))
                else:
                    print('WARNUNG: Kein Name für PID {} gefunden!'.format(pid))
                    out_lines.append('$       Shell element data for family    UNKNOWN_PID_{}\n'.format(pid))
                in_pshell_block = False
                pending_name    = None
                skip_continuation = True
                continue
            in_pshell_block = False
            pending_name    = None

        # ── Restliche Abaqus-Noise loeschen ─────────────────────────────────
        if is_abaqus_noise(line):
            continue

        out_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    print('\nFertig! Ausgabe: ' + output_path)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Verwendung: python bdf_rewrite.py input.bdf output.bdf')
        sys.exit(1)
    process_bdf(sys.argv[1], sys.argv[2])