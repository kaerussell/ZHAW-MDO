import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# Alle .dat Dateien finden
dat_files = sorted(glob.glob("opt_scripts/combined/output/vlm_out/**/*.dat", recursive=True))

# Nur Lift-Distribution Dateien (nicht alle .dat)
lift_files = [f for f in dat_files if "lift" in f.lower()]

def parse_dat(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) == 4:
                    data.append(vals)
            except:
                continue
    return np.array(data)

# Alle Dateien einlesen
all_data = []
all_labels = []
for f in lift_files:
    d = parse_dat(f)
    if len(d) > 0:
        all_data.append(d)
        all_labels.append(os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f))

if not all_data:
    print("Keine Dateien gefunden!")
    exit()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

d = all_data[0]
line_lift, = ax.plot(d[:,0], d[:,2], 'b-', lw=2, label="Normalized Lift")
line_ell,  = ax.plot(d[:,0], d[:,3], 'k--', lw=1, label="Elliptical")

ax.set_xlabel("η (spanwise)")
ax.set_ylabel("Normalized Lift")
ax.set_xlim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
title = ax.set_title(all_labels[0])

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
slider = Slider(ax_slider, "Iteration", 0, len(all_data)-1,
                valinit=0, valstep=1)

def update(val):
    i = int(slider.val)
    d = all_data[i]
    line_lift.set_ydata(d[:,2])
    line_ell.set_ydata(d[:,3])
    title.set_text(all_labels[i])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()