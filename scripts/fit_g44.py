import numpy as np


def corrected_debye_from_scales(
    Cp_GPa,
    C44_GPa,
    B0_GPa,
    density,
    a0_bohr,
    atoms_per_cell,
    sCp=1.0,
    s44=1.0,
    sB=1.0,
):
    """
    Recompute Debye temperature after empirical scaling of elastic constants.

    Cp_GPa  = C' = (C11 - C12) / 2
    C44_GPa = C44
    B0_GPa  = bulk modulus

    Scaling:
        Cp_used  = sCp * Cp_GPa
        C44_used = s44 * C44_GPa
        B_used   = sB * B0_GPa
    """

    Cp = sCp * Cp_GPa
    C44 = s44 * C44_GPa
    B0 = sB * B0_GPa

    # Mechanical sanity check
    if Cp <= 0 or C44 <= 0 or B0 <= 0:
        return np.nan, np.nan, Cp, C44, B0

    # Voigt-Reuss-Hill shear modulus for cubic system
    Gv = (2.0 * Cp + 3.0 * C44) / 5.0
    Gr = 10.0 * Cp * C44 / (4.0 * C44 + 6.0 * Cp)
    Gh = 0.5 * (Gv + Gr)

    if Gh <= 0:
        return np.nan, Gh, Cp, C44, B0

    # SI units
    B0si = B0 * 1.0e9
    Ghsi = Gh * 1.0e9

    bohr_to_m = 5.29177210903e-11
    volsi = (a0_bohr * bohr_to_m) ** 3

    vt = np.sqrt(Ghsi / density)
    vl = np.sqrt((B0si + 4.0 * Ghsi / 3.0) / density)

    vm = (3.0 / (2.0 / vt**3 + 1.0 / vl**3)) ** (1.0 / 3.0)

    theta = (
        4.79924e-11
        * (3.0 * atoms_per_cell / (4.0 * np.pi * volsi)) ** (1.0 / 3.0)
        * vm
    )

    return theta, Gh, Cp, C44, B0


# -------------------------------------------------------------------------
# Reference data
# -------------------------------------------------------------------------

reference = [
    # name, Cp_GPa, C44_GPa, B0_GPa, density, a0_bohr, theta_exp
    ("V",  91.3899, 118.4626, 176.703,  6110.0, 5.667246,  380.0),
    ("Mo", 186.0027, 235.7137, 240.792, 10220.0, 5.9915005, 450.0),
    ("W",  204.3522, 321.1925, 326.932, 19300.0, 5.9993161, 400.0),
    # Add Ta if desired:
    # ("Ta", 114.1732, 248.3889, 197.205, 15578.0, 6.2605805, 245.0),
]

atoms_per_cell = 2


# -------------------------------------------------------------------------
# Choose what to fit
# -------------------------------------------------------------------------

# Option A: fit only C44 scale
sCp_grid = [1.0]
s44_grid = np.linspace(0.20, 1.00, 161)
sB_grid = [1.0]

# Option B: fit Cp and C44 scales
# sCp_grid = np.linspace(0.50, 1.50, 101)
# s44_grid = np.linspace(0.20, 1.00, 161)
# sB_grid = [1.0]

# Option C: fit Cp, C44, and B0 scales
# sCp_grid = np.linspace(0.50, 1.50, 101)
# s44_grid = np.linspace(0.20, 1.00, 161)
# sB_grid = np.linspace(0.80, 1.20, 81)


# -------------------------------------------------------------------------
# Grid search
# -------------------------------------------------------------------------

best = None
all_results = []

for sCp in sCp_grid:
    for s44 in s44_grid:
        for sB in sB_grid:
            rel_errors = []
            rows = []

            for name, Cp, C44, B0, density, a0, theta_exp in reference:
                theta_pred, Gh_corr, Cp_corr, C44_corr, B0_corr = corrected_debye_from_scales(
                    Cp_GPa=Cp,
                    C44_GPa=C44,
                    B0_GPa=B0,
                    density=density,
                    a0_bohr=a0,
                    atoms_per_cell=atoms_per_cell,
                    sCp=sCp,
                    s44=s44,
                    sB=sB,
                )

                if not np.isfinite(theta_pred):
                    rel_errors.append(np.inf)
                    continue

                rel_error = theta_pred / theta_exp - 1.0
                rel_errors.append(rel_error ** 2)

                rows.append(
                    {
                        "name": name,
                        "theta_exp": theta_exp,
                        "theta_pred": theta_pred,
                        "rel_error_percent": 100.0 * rel_error,
                        "Gh_corr_GPa": Gh_corr,
                        "Cp_corr_GPa": Cp_corr,
                        "C44_corr_GPa": C44_corr,
                        "B0_corr_GPa": B0_corr,
                    }
                )

            rmse = np.sqrt(np.mean(rel_errors))

            result = {
                "rmse_relative": rmse,
                "sCp": float(sCp),
                "s44": float(s44),
                "sB": float(sB),
                "rows": rows,
            }

            all_results.append(result)

            if best is None or rmse < best["rmse_relative"]:
                best = result


# -------------------------------------------------------------------------
# Print best result
# -------------------------------------------------------------------------

print("Best fit:")
print(f"  relative RMSE = {best['rmse_relative']:.6f}")
print(f"  sCp = {best['sCp']:.6f}")
print(f"  s44 = {best['s44']:.6f}")
print(f"  sB = {best['sB']:.6f}")

print()
print("Per-element results:")
for row in best["rows"]:
    print(
        f"{row['name']:>2s} | "
        f"theta_exp={row['theta_exp']:8.2f} K | "
        f"theta_pred={row['theta_pred']:8.2f} K | "
        f"error={row['rel_error_percent']:8.2f}% | "
        f"Cp={row['Cp_corr_GPa']:8.2f} GPa | "
        f"C44={row['C44_corr_GPa']:8.2f} GPa | "
        f"Gh={row['Gh_corr_GPa']:8.2f} GPa | "
        f"B0={row['B0_corr_GPa']:8.2f} GPa"
    )