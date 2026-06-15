"""Plot the Lee-Yang E/N(x) benchmark (paper Eq. 1) for the dilute Bose gas.

E/N is in units of hbar^2/(2 m a^2) = 1/a^2. Shows the full Lee-Yang result and
the leading mean-field term 4*pi*x, so the onset of the LHY sqrt(x) correction is
visible. Run from this directory:  uv run python plot_lee_yang.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from dilute_gas import lee_yang_energy_per_particle

x = np.logspace(-6, -1.5, 300)  # gas parameter rho a^3
ly = np.array([lee_yang_energy_per_particle(xi, a=1.0) for xi in x])
mf = 4.0 * math.pi * x  # mean-field leading order
lhy_fraction = (128.0 / 15.0) * np.sqrt(x / math.pi)  # relative size of correction

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# E/N vs x
ax.loglog(x, ly, lw=2, label=r"Lee–Yang  $\frac{E}{N}=4\pi x\,[1+\frac{128}{15}\sqrt{x/\pi}]$")
ax.loglog(x, mf, "--", lw=1.3, color="0.5", label=r"mean field  $4\pi x$")
ax.set_xlabel(r"gas parameter  $x=\rho a^3$")
ax.set_ylabel(r"$E/N$  [$\hbar^2/2ma^2 = 1/a^2$]")
ax.set_title("Lee–Yang energy per particle (3D dilute Bose gas)")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.25)

# relative LHY correction
ax2.semilogx(x, 100 * lhy_fraction, lw=2, color="C3")
ax2.axhline(0, color="k", lw=0.6)
for xm in (1e-3, 1e-2):
    ax2.axvline(xm, color="0.7", ls=":", lw=1)
ax2.set_xlabel(r"gas parameter  $x=\rho a^3$")
ax2.set_ylabel(r"LHY correction  $\frac{128}{15}\sqrt{x/\pi}$  [%]")
ax2.set_title("Beyond-mean-field correction vs $x$")
ax2.grid(True, which="both", alpha=0.25)

fig.tight_layout()
out = "lee_yang.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"  x=1e-4:  E/N={lee_yang_energy_per_particle(1e-4):.4e}   LHY corr={100*(128/15)*math.sqrt(1e-4/math.pi):.2f}%")
print(f"  x=1e-3:  E/N={lee_yang_energy_per_particle(1e-3):.4e}   LHY corr={100*(128/15)*math.sqrt(1e-3/math.pi):.2f}%")
print(f"  x=1e-2:  E/N={lee_yang_energy_per_particle(1e-2):.4e}   LHY corr={100*(128/15)*math.sqrt(1e-2/math.pi):.2f}%")
