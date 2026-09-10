import numpy as np
import json
import os
import pandas as pd
from src.features import compute_hea_features
from src.consts import composition_labels, PHYSICAL_CP_MIN_GPA, PHYSICAL_THETA_MIN_K, ATOMS_PER_CELL, DG_T_K, STONER_I_EV, E_SF_MEV, C_REF_CP_GPA, STONER_S_REF, STONER_MU_BASE
from src.elements import ELEMENTS


def normalize_composition(composition):
    composition = np.array(composition, dtype=float)
    total = np.sum(composition)
    if total <= 0:
        raise ValueError("Invalid composition sum")
    return (composition / total).tolist()


_Ry_to_eV = 13.605693123
_kB_eV = 8.617333262e-5   # eV/K
_Ry_to_J = 2.1798741e-18
_Bohr_to_m = 5.29177210903e-11
_amu_to_kg = 1.66053906660e-27
_hbar = 1.054571817e-34
_kB = 1.380649e-23
# C such that λ = eta [Ry/Bohr²] * C / (M_mix [amu] * Theta_D [K]²)
# Derived from Debye model: <ω²> = (3/5)(k_B Θ_D/ħ)², validated on Ta without fitting
_C_THEORETICAL = (5.0/3.0) * (_Ry_to_J / _Bohr_to_m**2) * _hbar**2 / (_amu_to_kg * _kB**2)


def compute_lambda(row):
    used_labels = [e for e in composition_labels if e in row.keys()]
    nominator = np.sum([row[e]*row[f'{e}_eta_total'] for e in used_labels])
    mixture_mass = row['mixture_mass']
    denominator = mixture_mass*row['thetaDB']**2
    return nominator/denominator*_C_THEORETICAL

def compute_lambda_nocutoff(row):
    used_labels = [e for e in composition_labels if e in row.keys()]
    nominator = np.sum([row[e]*row[f'{e}_eta_total_full'] for e in used_labels])
    mixture_mass = row['mixture_mass']
    denominator = mixture_mass*row['thetaDB']**2
    return nominator/denominator*_C_THEORETICAL

def read_params(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    concentrations = normalize_composition(data['concentrations'])
    for l, c in zip(data['element_labels'], concentrations):
        data[l] = c
    del data['element_labels']
    del data['concentrations']
    return data

def read_debye(path, dirname):
    data = json.load(open(os.path.join(path, dirname, "debye", "debye_summary.json"), "r"))

    # Backward compatibility:
    # old code downstream expects "thetaDB"
    if "thetaDB_K" in data and "thetaDB" not in data:
        data["thetaDB"] = data["thetaDB_K"]

    # Remove file/log metadata if present
    for key in [
        "run_log",
        "results_csv",
        "all_scf_results_csv",
        "even_energy_fit_data_csv",
        "rmt_candidates_json",
    ]:
        data.pop(key, None)

    # Remove large nested fit diagnostics if you do not want them as ML columns
    for key in [
        "tetra_fit",
        "c44_fit",
        "C44_diagnostics_Ry_bohr3",
        "C44_diagnostics_GPa",
    ]:
        data.pop(key, None)

    # Remove old single-run energy keys if present
    for key in [
        "delta",
        "energy0_mono_ev",
        "energy0_tetra_ev",
        "energy_tetra",
        "energy_mono",
    ]:
        data.pop(key, None)

    return data

def get_composition(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    return normalize_composition(data['concentrations']), data['element_labels']

def read_macmillan(path, dirname):
    composition_dict, elements = get_composition(path, dirname)
    df = pd.read_csv(open(os.path.join(path, dirname, 'finalscf', 'mcmillan_cutoff_results.csv')))
    df = df[(df['reduce_mode'] == 'mean') & (df['integral_mode'] == 'plain') & (df['norm_mode'] == 'none')]
    assert len(df) == len(composition_dict)
    df['component_label'] = [elements[x-1] for x in df['component']]
    df.reset_index(drop=True, inplace=True)
    results = {}
    for _, row in df.iterrows():
        cmp_label = row['component_label']
        for ch in ['sp', 'pd', 'df']:
            # cutoff (primary)
            results[f'{cmp_label}_eta_{ch}'] = row[f'eta_cutoff_{ch}']
            results[f'{cmp_label}_M_{ch}'] = row[f'M_cutoff_{ch}']
            # full integral (no cutoff)
            results[f'{cmp_label}_eta_{ch}_full'] = row[f'eta_full_{ch}']
            results[f'{cmp_label}_M_{ch}_full'] = row[f'M_full_{ch}']
        results[f'{cmp_label}_eta_total'] = row['eta_total_cutoff']
        results[f'{cmp_label}_eta_total_full'] = row['eta_total_full']
        results[f'{cmp_label}_Ntot'] = row['Ntot']  # states/Ry/atom/spin; used for Stoner correction
    return results

def tc_from_data(data, mu):
    return data['thetaDB']/1.45*np.exp(-1.04*(1+data['lambda'])/(data['lambda']-mu*(1+0.62*data['lambda'])))

def _tc_mcmillan_safe(theta_D, lam, mu):
    """McMillan formula; returns 0.0 when mu* is too large to allow pairing."""
    denom = lam - mu * (1.0 + 0.62 * lam)
    if denom <= 0.0:
        return 0.0
    return theta_D / 1.45 * np.exp(-1.04 * (1.0 + lam) / denom)

def compute_stoner_correction(data, composition_dict, mu_star_coulomb=0.13):
    """
    Calibrated spin-fluctuation correction to Tc via Berk-Schrieffer (1966).

    Uses the excess-correction method calibrated 2026-09-10 against 36 literature
    baseline alloys (results/Tc_baseline.xlsx).  Only the EXCESS spin-fluctuation
    pair-breaking above a reference Stoner level S_REF is added to a calibrated
    base mu* (STONER_MU_BASE), so that alloys with typical S ~ S_REF are unaffected
    while high-S alloys (Ti/Sc-rich) receive a composition-specific downward
    correction.  Calibration gives median Tc_sf/Tc_exp = 1.00, rms(log) = 0.51
    over the 36-alloy baseline (vs 1.60x without correction, 0.59x with full
    Berk-Schrieffer).  See STONER_CORRECTION.md and consts.py for details.

    gamma = ln(E_sf_mix / omega_D) uses composition-weighted element E_sf from
    E_SF_MEV (Option B: no fitting, fully predictive).
    N_i(EF) from KKR Ntot (states/Ry/atom/spin), converted to states/eV.

    Returns a dict with all intermediate quantities plus Tc_sf.
    """
    total_c = sum(c for c in composition_dict.values() if c > 0)
    if total_c <= 0:
        return {}

    IN_mix = 0.0
    E_sf_mix = 0.0  # meV
    for el, c in composition_dict.items():
        if c <= 0:
            continue
        ntot = data.get(f'{el}_Ntot')
        if ntot is None:
            continue
        I_i   = STONER_I_EV.get(el, 0.40)        # eV
        N_i   = ntot / _Ry_to_eV                  # states/eV/atom/spin
        E_sf_i = E_SF_MEV.get(el, 100.0)          # meV
        frac  = c / total_c
        IN_mix   += frac * I_i * N_i
        E_sf_mix += frac * E_sf_i

    # Guard: Stoner instability → ferromagnetic, formula invalid
    IN_mix = min(IN_mix, 0.99)
    S = 1.0 / (1.0 - IN_mix)
    lambda_sf = S - 1.0   # = IN_mix / (1 - IN_mix)

    theta_D   = data.get('thetaDB', 0.0)
    omega_D_meV = theta_D * _kB_eV * 1000.0  # K → meV  (kB in eV/K, ×1000 → meV/K)

    # Option B+ (C'-dependent γ): scale E_sf_mix by instability proximity
    E_sf_eff = E_sf_mix
    if C_REF_CP_GPA is not None:
        cp_gpa = data.get('Cp_GPa', C_REF_CP_GPA)
        instability_factor = min(1.0, cp_gpa / C_REF_CP_GPA)
        E_sf_eff = E_sf_mix * instability_factor

    if omega_D_meV > 0 and E_sf_eff > 0:
        gamma = np.log(E_sf_eff / omega_D_meV)
    else:
        gamma = 0.0

    # Full Berk-Schrieffer mu_sf (stored for diagnostics)
    mu_sf_full = lambda_sf / (1.0 + lambda_sf * gamma)

    # Excess correction: subtract the baseline spin-fluctuation contribution at S_REF
    # (using the same composition-specific gamma so the reference is on the same scale).
    # mu_sf_excess = 0 when S <= S_REF; grows for S >> S_REF.
    lsf_ref   = STONER_S_REF - 1.0
    mu_sf_ref = lsf_ref / (1.0 + lsf_ref * gamma) if (1.0 + lsf_ref * gamma) > 0 else 0.0
    mu_sf_excess = max(0.0, mu_sf_full - mu_sf_ref)

    # Calibrated effective mu*: STONER_MU_BASE absorbs Coulomb (0.13) + baseline sf.
    # Only excess pair-breaking beyond S_REF is added.
    mu_eff = STONER_MU_BASE + mu_sf_excess

    lam    = data.get('lambda', 0.0)
    Tc_sf  = _tc_mcmillan_safe(theta_D, lam, mu_eff)

    return {
        'Stoner_IN_mix':       IN_mix,
        'Stoner_S':            S,
        'Stoner_lambda_sf':    lambda_sf,
        'Stoner_gamma':        gamma,
        'Stoner_E_sf_meV':     E_sf_mix,
        'Stoner_E_sf_eff_meV': E_sf_eff,
        'Stoner_mu_sf':        mu_sf_full,
        'Stoner_mu_sf_ref':    mu_sf_ref,
        'Stoner_mu_sf_excess': mu_sf_excess,
        'Stoner_mu_eff':       mu_eff,
        'Tc_sf':               Tc_sf,
    }

def process_kkr(path, dirname):
    try:
        composition, elements = get_composition(path, dirname)
        comp_dict = dict(zip(elements, composition))
        # Mind: dict merging old-style as this has to run on old python
        #data = {'name': dirname} | read_params(path, dirname) | read_debye(path, dirname) | read_macmillan(path, dirname)
        data = {'name': dirname}
        data.update(read_params(path, dirname))
        data.update(read_debye(path, dirname))
        data.update(read_macmillan(path, dirname))
        if data.get('use_mixture_debye'):
            data['thetaDB'] = data['mixture_debye_temperature']
        data['lambda'] = compute_lambda(data)
        cp_ok = data.get('Cp_GPa', 0.0) >= PHYSICAL_CP_MIN_GPA
        theta_ok = data.get('thetaDB', 0.0) >= PHYSICAL_THETA_MIN_K
        data['outside_range'] = not (cp_ok and theta_ok)
        if data['outside_range']:
            data['lambda_unphysical'] = data['lambda']
            data['lambda'] = 0.0
            data['Tc_mu0.1'] = 0.0
            data['Tc_mu0.2'] = 0.0
            data['Tc_mu0.3'] = 0.0
        else:
            data['Tc_mu0.1'] = tc_from_data(data, mu=0.1)
            data['Tc_mu0.2'] = tc_from_data(data, mu=0.2)
            data['Tc_mu0.3'] = tc_from_data(data, mu=0.3)
        data['lambda_nocutoff'] = compute_lambda_nocutoff(data)
        _lam_nc = data['lambda_nocutoff']
        data['Tc_mu0.1_nocutoff'] = tc_from_data({**data, 'lambda': _lam_nc}, mu=0.1)
        data['Tc_mu0.2_nocutoff'] = tc_from_data({**data, 'lambda': _lam_nc}, mu=0.2)
        data['Tc_mu0.3_nocutoff'] = tc_from_data({**data, 'lambda': _lam_nc}, mu=0.3)
        # Stoner spin-fluctuation correction (Option B)
        stoner = compute_stoner_correction(data, comp_dict)
        data.update(stoner)
        if data['outside_range']:
            data['Tc_sf'] = 0.0
        # add features
        data = {**data, **compute_hea_features(comp_dict=comp_dict, normalize_composition=True)}

        # thermodynamic stability: dG = dE - T*dS_conf  (eV/atom, w.r.t. pure BCC phases)
        if 'energy0_Ry' in data and 'config_entropy_nat' in data:
            energy_per_atom_Ry = data['energy0_Ry'] / ATOMS_PER_CELL
            ebulk_mix = sum(comp_dict.get(el, 0.0) * ELEMENTS[el].ebulk for el in comp_dict if el in ELEMENTS)
            dE_eV = (energy_per_atom_Ry - ebulk_mix) * _Ry_to_eV
            data['dG_eV'] = dE_eV - DG_T_K * _kB_eV * data['config_entropy_nat']

        return data
    except Exception as e:
         print(f'Error in processing {dirname}: {e}')
         return None

#out = process_kkr(path='/home/rafal/WORK/HEA/ML/random.ratios/sra.kp10.ew0.6/', dirname='Ti0.0008Nb0.3225Zr0.0191Hf0.4401Ta0.0272Sc0.0316Mo0.0243W0.0594Y0.0584La0.0165')
#out = process_kkr(path='/home/rafal/WORK/HEA/', dirname='TEMP')
