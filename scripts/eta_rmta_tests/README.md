# η (McMillan–Hopfield) in RMTA — tests against Jaśkiewicz 2016 / Rajput 1996

Session 2026-10-22. Question: our `macmillan_cutoff.py` integrates
M = ∫ u_l (dV/dr) u_{l+1} dr from the *last node* of the wavefunction (heuristic), because the
full integral from r = 0 explodes (η_sp ≈ 41 000 mRy/a₀² for Ta). Is the cutoff justified, and why
do Jaśkiewicz et al. (same formula, Eq. 2) get small η_sp with the full integral?

All numbers below: Ta34Nb33Hf8Zr14Ti11, `results/comparison.sra.ew0.6.bz10-10-20/Ta34Nb33Hf8Zr14Ti11`
(SRA, PBE, ew 0.6, mxl 3, final SCF bzqlty 20, a0 = 6.3588 bohr = 3.365 Å; Jaśkiewicz: a = 3.36 Å).
η in mRy/a₀², sp / pd / df / total. Reference: Jaśkiewicz 2016 Table 1 (`papers_data/Jaskiewicz2016.pdf`),
Rajput 1996 Table 8 (`papers_data/S_S_Rajput_1996_...pdf`, 1 eV/Å² = 20.58 mRy/a₀²).

## Scripts (run from repo root; each takes a process_hea workdir or a fort.51[.gz])

| script | what it does |
|---|---|
| `01_gg_identity.py <workdir>` | on the AkaiKKR mesh: V(r) vs −2Z/r near the nucleus, small-r power law of u_l (nonrel l+1 vs SRA γ), and the Gaspari–Gyorffy identity check: M(r₀→R) numeric vs surface term S(R)−S(r₀) for r₀ = 10⁻⁶…1 a₀ |
| `02_resolve_radial.py <workdir>` | re-solves the radial equation (nonrel and SRA/Koelling–Harmon) in the fort.51 potential at E_F on a 40 000-pt mesh; full integral vs surface formula for both; compares with the AkaiKKR functions (Ta, Ti) |
| `03_eta_table.py <workdir> [...]` | per-element N(E_F) and η: full integral (production `_full`), surface formula, last-node cutoff (`max` mode) vs Jaśkiewicz Table 1 |
| `04_surface_pure_elements.py` | surface-formula η for pure Ta, Nb (`results/.../Ta`, `Nb`) vs Jaśkiewicz Table 2 and λ_exp; E_F sensitivity of Ta channels |
| `05_ws_normalization.py` | continues u_l beyond R_MT with free solutions (constant V), renormalizes to the Wigner–Seitz sphere, rescales η by f_l·f_{l+1}, f_l = N_MT/N_WS |
| `common.py` | fort.51(.gz) loading, labels from run_params.json |

`nrl` test run (final SCF only, Jaśkiewicz lattice constant, 5-component CPA, ~10 min local):
```
python scripts/finalscf.py --output finalscf --elements 73 41 72 40 22 --concentrations 0.34 0.33 0.08 0.14 0.11 \
  --a0 6.34949 --ew 0.6 --xc pbe --rel nrl --bzqlty 20 --mxl 3 --workdir results/test.nrl.ew0.6/TNHZT_a3.36_nrl --no-gzip
```
(`finalscf.py` called directly needs atomic numbers and fractional concentrations; fort.51 format does not depend on reltyp.)
Result kept in `results/test.nrl.ew0.6/TNHZT_a3.36_nrl/`.

## Surface (Gaspari–Gyorffy) formula

For solutions of the **nonrelativistic** radial equation at the same energy E,
V′ = P·H_l − H_{l+1}·P with P = d/dr − (l+1)/r, hence the whole integral equals a boundary term:

M = u_{l+1}(R)·[(V(R) − E + l(l+1)/R²)·u_l(R) + (l+1)·u_l(R)/R² − (l+1)·u_l′(R)/R] − u′_{l+1}(R)·[u_l′(R) − (l+1)·u_l(R)/R]

(implemented in 02–05; verified: for nonrel solutions full integral = surface term to all printed digits).

## Results

### 1. Why the full integral explodes (SRA, AkaiKKR mesh)
| r₀ [a₀] | 10⁻⁶ | 10⁻³ | 10⁻² | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|---|
| Ta sp: M(r₀→R) numeric | 51.7 | 41.8 | −55.5 | −37.3 | −14.9 | −1.71 |
| Ta sp: S(R)−S(r₀) | 0.4 | −33.8 | −80.6 | −35.8 | −14.8 | −1.71 |
No plateau: a fixed lower limit is as arbitrary as the last node. Agreement only for r₀ ≳ 0.3 a₀.

### 2. Dense-mesh re-solve (02)
| | Ti sp/pd/df | Ta sp/pd/df |
|---|---|---|
| Jaśkiewicz 2016 | 4.9 / 46 / 76 | 0.9 / 51 / 102 |
| full integral, AkaiKKR mesh (SRA, = production `_full`) | 48 / 26 / 70 | 41 433 / 148 / 104 |
| full integral, dense mesh, SRA functions | 5.9 / 33 / 70 | 1 875 / 1 159 / 125 |
| full integral, dense mesh, nonrel functions | 3.3 / 30 / 70 | 17 / 180 / 103 |
| **surface formula, SRA (AkaiKKR) functions** | 3.0 / 29 / 70 | 4.5 / 120 / 111 |
| last-node cutoff (production) | 7 / 110 / 70 | 13 / 159 / 61 |
- Ti: the 48 was purely numerical (mesh + np.gradient); cutoff is wrong for Ti (pd 110 vs ~30).
- Ta: with SRA functions the volume integral is ill-defined near the nucleus (orthogonality + (Z/r)/c²); the surface term is the only stable, parameter-free quantity. Dense-mesh SRA solve reproduces AkaiKKR u to 0.5 %.

### 3. `nrl` final SCF at a = 3.36 Å (03)
- Full integral = surface term to 3 decimals for every element/channel: Eq. (2) is well-defined only for nonrelativistic functions (Rajput 1996 Eq. 4: "regular solutions of the Schrödinger equation").
- N(E_F) per element (both spins, 1/Ry): Jaśkiewicz 17.2/18.8/16.2/17.2/26.6; ours SRA 16.4/18.0/15.4/16.9/25.3; **ours nrl 17.1/19.0/16.2/17.2/25.1** → their electronic structure looks nonrelativistic.
- η nrl (Ta/Nb/Hf/Zr/Ti): 20/123/116=259, 17/96/110=223, 29/182/102=313, 30/161/99=289, 4.5/30/69=103 → the pd excess (~2× for 4d/5d) is **not** relativistic.

### 4. Wigner–Seitz normalization (05), SRA alloy
| | f_s f_p f_d f_f | η MT-norm sp/pd/df | η WS-norm sp/pd/df | Jaśkiewicz |
|---|---|---|---|---|
| Ta | .73 .56 .86 .42 | 4.5/120/111 | 1.8/58/40 | 0.9/51/102 |
| Nb | .67 .55 .88 .42 | 10/99/109 | 3.8/48/40 | 4.7/55/102 |
| Hf | .70 .55 .81 .44 | 8/187/91 | 3.1/84/32 | 1.7/71/88 |
| Zr | .64 .55 .83 .43 | 18/172/98 | 6.5/79/35 | 7.1/79/90 |
| Ti | .68 .57 .89 .38 | 3/29/70 | 1.2/15/24 | 4.9/46/76 |
Pure Nb: MT 13/89/93 (=195), WS 4.7/43/34; Rajput 7.5/62/98 (=167.5), Papaconstantopoulos 157.
- **sp and pd of Jaśkiewicz are reproduced by WS-sphere normalization** (Zr pd 79.2 vs 79.3) — the 2× pd discrepancy is the normalization-sphere convention of RMTA (Rajput Eq. 4: "normalized to unity in the Wigner–Seitz sphere"), not an error.
- **df matches only with MT normalization** (WS would give ~40 vs their ~100). With "s,p,d in WS, f in MT": Ta 154.5 (154.0), Hf 160.8 (160.6), Nb 147 (162), Zr 167 (176), pure Ta 149 (155). Empirical; why l=3 differs in their code is unknown.
- Ti fits in no convention (pd 15–29 vs 46): open.

## Conclusions
1. Our η is a correct RMTA (surface/GG form, MT normalization, SRA functions); df agrees with Kraków to ~10 %, sp/pd differ by the sphere convention (×0.5–0.6).
2. Absolute η in RMTA is ambiguous by ~1.5× (sphere, ⟨ω²⟩ = ½ vs ⅗ θ², θ_D source) — with Jaśkiewicz's η_mix (157) and our conventions λ_mix = 0.70 instead of 1.04. Absolute λ is only meaningful together with the fitted μ* (ours 0.205; they also need ~0.2).
3. The last-node cutoff should be replaced by the surface formula (parameter-free; fixes Ti pd 110 → 29; Ta/Nb sums nearly unchanged; element weights change: Ti 187 → 102, Hf 161 → 286).
4. η_mix of the alloy: Jaśkiewicz 157 | surface 226 | cutoff 217 | nrl 239 → λ of this alloy barely changes; the ranking of alloys may (Ti- vs Hf-rich).

## Open / next steps
- Implement `--cutoff-mode surface` in `macmillan_cutoff.py` (needs only u, u′, V at R_MT and E_F from fort.51), recompute the 106 baseline alloys from local `fort.51.gz` (no KKR), compare τ/MAE/μ* with the cutoff version (`notebooks/Tc_Baseline_Comparison.ipynb`). Optimization data keep fort.51.gz → also recomputable.
- Pure Nb/Ta in nrl vs Jaśkiewicz Table 2 / Papaconstantopoulos 1977 (PRB 15, 4221).
- Ti pd channel (3d) unexplained.
- Production `np.gradient` for dV/dr adds ~7 mRy/a₀² in Ta sp (spline derivative removes it) — irrelevant once the surface form is used.

## One-table comparison of conventions (SRA functions, alloy; η mRy/a₀², sp/pd/df/total)

| | Jaśkiewicz | A: last-node cutoff (production) | B: surface, MT | C: surface, WS all l | D: sp,pd WS / df MT | E: s,p,d WS / f MT |
|---|---|---|---|---|---|---|
| Ta | 0.9/51/102/**154** | 13/159/61/**232** | 4.5/120/111/**236** | 1.8/58/40/**99** | 1.8/58/111/**170** | 1.8/58/95/**154** |
| Nb | 4.7/55/102/**162** | 10/120/109/**239** | 10/99/109/**218** | 3.8/48/40/**92** | 3.8/48/109/**161** | 3.8/48/96/**147** |
| Hf | 1.7/71/88/**161** | 10/118/33/**161** | 8/187/91/**286** | 3.1/84/32/**119** | 3.1/84/91/**178** | 3.1/84/74/**161** |
| Zr | 7.1/79/90/**176** | 8/80/97/**185** | 18/172/98/**288** | 6.5/79/35/**120** | 6.5/79/98/**183** | 6.5/79/81/**167** |
| Ti | 4.9/46/76/**126** | 7/110/70/**187** | 3.0/29/70/**102** | 1.2/15/24/**40** | 1.2/15/70/**86** | 1.2/15/62/**78** |
| η_mix | **157** | 217 | 226 | 94 | 160 | 146 |
| λ_mix (θ_mix 276 K, ⅗θ²) | (1.16 with θ 243 K, ½θ²; 0.70 in our conventions) | 1.04 | 1.09 | 0.45 | 0.77 | 0.70 |
| Tc, μ* = 0.2 (exp 7.3 K) | 10.7 (2.6 ours) | 9.7 | 10.6 | 0.1 | 3.9 | 2.6 |

C excluded (df → 40). D/E hit their η_mix but not Ti, and need μ* ≈ 0.13–0.15 for 7.3 K. Convention choice
moves λ over 0.7–1.1 for the same calculation; for alloy ranking consistency matters, hence **B** is adopted
for the baseline recomputation (see `06_recompute_baseline_surface.py`).

## Baseline recomputed with the surface formula (06, 07) — 2026-10-22

`macmillan_cutoff.py` now has `--cutoff-mode surface` (boundary term, MT normalization; `compute_M_surface`,
E_F passed as `block["ef"]`). `06_recompute_baseline_surface.py` → `results/comparison.sra.ew0.6.bz10-10-20/eta_surface.csv`
(102/106; pure La, Re, Ti, Zr have `thetaDB_K = None`, not in the literature set).
`07_compare_fit_cutoff_vs_surface.py` (run from `notebooks/`; uses `baseline_loader.py` = the loader cell of
Tc_Baseline_Comparison.ipynb) → `fit_cutoff_vs_surface_rows.csv`. θ_D mix, λ_mix rescaling, μ* = 0.2, set B:

| metric (n=74 / bcc n=65) | A: last-node cutoff | B: surface |
|---|---|---|
| Kendall τ | 0.722 / 0.739 | 0.701 / 0.725 |
| Spearman ρ | 0.893 / 0.906 | 0.881 / 0.897 |
| Pearson r | 0.854 / 0.864 | **0.877 / 0.896** |
| MAE [K] | **1.95 / 1.96** | 2.09 / 2.10 |
| bias [K] | 1.48 / 1.46 | 1.68 / 1.69 |
| rms log | 0.327 / 0.331 | **0.309 / 0.306** |
| median Tc_KKR/Tc_exp | 1.22 | 1.28 |
| LR slope / intercept | 0.44 / 3.0 | 0.44 / 2.9 |
| pairs ΔTc_exp>1 K misordered | 4.9 % / 4.2 % | 4.6 % / 3.7 % |
| best μ* (abs) / rms log there | 0.205 / 0.321 | 0.210 / 0.292 |
| τ from λ_mix alone | 0.736 | 0.646 |

λ_B/λ_A over 65 alloys: 0.88–1.15 (median 1.02), Spearman with x_Ti = −0.93 (Ti-rich drop up to 12 %:
Ti50ZrNbHfTa50 0.88, NbTiZr 0.93), Hf/Ta/Mo/W-rich rise 10–15 % (Hf35Nb10Ta35Mo10W10 1.15, HfNbTa 1.10).
**Verdict: no clear winner.** B is better in r, rms log and misordered pairs (the Ti-driven change compresses
the λ-dependent residual: underestimated Mo/W/Hf alloys go up, overestimated Ti-rich go down), A is better in
τ/ρ/MAE by a similar small margin (Δτ ≈ 0.02 ≈ bootstrap noise). Physics argues for B (parameter-free,
correct for light elements); the fit does not discriminate. Decision pending.

### D/E hybrids: SRA vs NRL functions (same alloy; NRL = `results/test.nrl.ew0.6/TNHZT_a3.36_nrl`)

| | Jaśkiewicz | B SRA | B NRL | D SRA | D NRL | E SRA | E NRL |
|---|---|---|---|---|---|---|---|
| Ta | 0.9/51/102/**154** | 4.5/120/111/**236** | 20.5/123/116/**259** | 1.8/58/111/**170** | 7.3/59/116/**182** | 1.8/58/95/**154** | 7.3/59/101/**167** |
| Nb | 4.7/55/102/**162** | 10.2/98/109/**218** | 17.3/96/110/**223** | 3.8/48/109/**161** | 6.2/47/110/**163** | 3.8/48/96/**147** | 6.2/47/97/**150** |
| Hf | 1.7/71/88/**161** | 8.1/187/91/**286** | 29.4/182/102/**313** | 3.1/84/91/**178** | 10.2/83/102/**195** | 3.1/84/74/**161** | 10.2/83/85/**177** |
| Zr | 7.1/79/90/**176** | 18.1/172/97/**288** | 29.5/161/99/**289** | 6.5/79/97/**183** | 10.3/75/99/**184** | 6.5/79/81/**167** | 10.3/75/83/**168** |
| Ti | 4.9/46/76/**126** | 3.0/29/70/**102** | 4.5/30/69/**103** | 1.2/15/70/**86** | 1.7/15/69/**86** | 1.2/15/62/**78** | 1.7/15/61/**78** |
| η_mix | **157** | **226** | **238** | **160** | **166** | **146** | **152** |
| λ_mix (θ 276 K) | 0.75 | 1.09 | 1.14 | 0.77 | 0.80 | 0.70 | 0.73 |
| T_c μ*=0.2 [K] | 3.6 | 10.6 | 11.8 | 3.9 | 4.5 | 2.6 | 3.2 |
| T_c μ*=0.13 [K] | 7.5 | 15.7 | 17.0 | 7.9 | 8.6 | 6.1 | 6.9 |

Jaśkiewicz λ (1.16, θ 243 K, ½θ²) recomputed with our conventions: 0.70 → Tc(μ* 0.2) 2.6 K, Tc(μ* 0.13) 6.1 K; exp 7.3 K.

## Decision 2026-10-22 (end of session)
**Production stays with the last-node cutoff (`mcmillan_cutoff_mode = 'valence'`, variant A).** The surface
formula (B) is implemented and available (`--cutoff-mode surface`), the baseline comparison showed no
statistically significant difference in the fit to experiment (Δτ ≈ −0.02, Δr ≈ +0.02, ΔMAE ≈ +0.15 K),
and switching would require recomputing the random/optimization datasets. To be revisited later; everything
needed is here (scripts 01–07, `eta_surface.csv`, `fit_cutoff_vs_surface_rows.csv`, nrl run).
NRL vs SRA: only the sp channel differs (NRL 3–4× larger); Jaśkiewicz's sp matches our SRA+WS (D/E), their
N(E_F) matches NRL; Ti fits no variant.

## Integration by parts (08_integration_by_parts.py, 2026-10-23)
Idea: avoid the numerical dV/dr by moving the derivative onto u: M = [u_l u_{l+1} V]_0^R − ∫ V (u_l u_{l+1})′ dr
(boundary term small but kept; V(R) ≈ −0.15 Ry on the mesh after dropping the last point). Result on the AkaiKKR mesh:
- NRL functions: direct (spline V′) = by parts = surface term to 3 decimals in every channel (Ta sp 0.958/0.958/0.958);
  production `np.gradient` adds 5–15 % in sp only.
- SRA functions: by parts reproduces the *exploded* direct value (Ta sp 52.8 vs 51.5 direct vs 0.53 surface; Nb 11.2/11.6/0.78;
  Ti 2.03/2.00/0.50) and for pd the three forms all differ (Ta 0.50 / 0.72 / −0.67).
Conclusion: the explosion is not a derivative artefact — the SRA radial functions themselves carry non-cancelling contributions
from r < 0.01 a₀ (cumulative by-parts integrand for Ta sp: 19 → 176 → 50 → 84 → 59 → 52 at r = 1e-3…1 a₀; NRL: 2 → 99 → 0.2 → 41 → 11 → −0.6, cancelling to 0.7).
Only the surface form (or genuinely nonrelativistic functions) gives a well-defined M.

## LDA (von Barth–Hedin, `--xc vbh`, as in Jaśkiewicz) vs PBE — 2026-10-23
Final SCF of the alloy at a = 3.36 Å, bz20, SRA and NRL: `results/test.nrl.ew0.6/TNHZT_a3.36_{sra,nrl}_vbh`.
E_F: PBE 0.657 → LDA 0.637 Ry. N(E_F) per element (both spins): LDA-NRL 17.2/19.1/16.4/17.3/25.5 vs Jaśkiewicz 17.2/18.8/16.2/17.2/26.6
(best match of all four combinations). η (surface, MT) changes by ≤ 5 % (pd −3…−5 %, df +3…+5 %):
B: LDA-SRA Ta 4.0/116/116=236, Nb 9.6/95/114=218, Hf 7.2/177/93=277, Zr 16.9/164/102=283, Ti 2.6/26/74=103; η_mix 225 (PBE 226).
D/E with LDA: η_mix 163–166 / 148–152 (PBE 160–166 / 146–152). **The functional is not the source of the sp/pd/Ti discrepancy.**
Ti pd stays 26–30 vs their 46 in every combination (PBE/LDA × SRA/NRL × B/D/E).

## Pure elements vs Papaconstantopoulos et al. PRB 15, 4221 (1977) — 2026-10-23 (09_pure_elements_vs_papaconstantopoulos.py)
Reference: nonrelativistic APW, GG phase-shift form, bcc for Ti? (Table I), Zr (a 6.60), Nb (6.20), Mo (5.89), Y; 1 eV/Å² = 20.58 mRy/a₀².
Ours: results/comparison.sra.ew0.6.bz10-10-20/<El> (SRA, PBE, bz20, equilibrium a0), surface formula; D = sp,pd × f_s f_p, f_p f_d (WS), df MT.
| El | Papa sp/pd/df/tot | ours B (MT) | ours D | phase shifts δ_s,p,d (Papa / ours) |
|---|---|---|---|---|
| Ti | 2.7/29/68/**100** | 3.4/63/69/136 | 1.3/32/69/**103** | −0.64,−0.16,0.75 / −0.82,−0.25,0.71 |
| Zr | 5.1/44/47/**96** | 5.9/106/68/180 | 2.2/49/68/**119** | (a0 6.746 vs 6.60) |
| Nb | 9.7/56/91/**157** | 12.8/89/93/195 | 4.7/43/93/**141** | −0.93,−0.36,1.14 / −1.03,−0.45,1.06 |
| Mo | 1.4/21/97/**119** | 4.7/35/104/143 | 1.8/18/104/**124** | −0.97,−0.38,−1.27 / −1.05,−0.45,−1.47 |
| Y | 1.0/21/26/**48** | 1.1/69/29/100 | 0.4/31/29/**61** | −0.70,−0.26,0.47 / −0.82,−0.34,0.49 |
df agrees with the MT form (B) within 10 % (Zr: lattice constant differs); pd of B is 1.6–3× too large; D reproduces Papa for Ti, Mo
(and Nb/Y within 25 %). Same pattern as vs Jaśkiewicz → two independent GG-based references say our MT-normalised pd is ~2× high.
GG 1972 text: R_l "scattering solution … for r > a (MT radius) R_l = j_l cosδ_l − n_l sinδ_l", n_l defined with ∫_0^{a_WS} r² R_l² (WS radius);
matrix elements of Eq. (5) "are equal to sin(δ_{l+1} − δ_l)". I.e. the GG normalisation is free-wave/WS-cell, not unit-in-MT; the exact
bookkeeping for AkaiKKR (whose dosef are full-cell partial DOS) still has to be derived — open.
Our Nb N(E_F) 13.9/Ry (both spins) vs Papa 2×9.71 = 19.4: pure Nb at bz20 is not k-converged (see k-point test).

## GG Eq. (7) evaluated directly (10_gg_phase_shift_form.py) — 2026-10-23
GG 1972 Eq. (5)–(7): η = (k²/π²N) Σ 2(l+1) sin²(δ_{l+1}−δ_l) (n_l/n_l^(1))(n_{l+1}/n_{l+1}^(1)), R_l free-wave normalised
(j cosδ − y sinδ outside R_MT), n_l^(1)(a) = (k/π)(2l+1)∫_0^a r²R_l² (single scatterer, same sphere as n_l).
Results (all elements, all channels): **M_free / sin(δ_{l+1}−δ_l) = −1.0000** → our surface formula IS the GG identity;
**GG(7) with MT-sphere ratios = variant B exactly; with WS-sphere ratios = variant C exactly.** D/E are not GG forms.
Papaconstantopoulos Table III is reproduced to 0.1 mRy/a₀² from their Tables II (δ_l, n_l/n_l^(1)) + IV (E_F, n(E_F)) with
E_F from the MT zero and N per spin → same formula and same (MT) convention as ours.
Factor decomposition ours/Papa (pure, bz20, PBE-SRA): Nb pd 1.59 = (k²/N 1.63)×(r_p r_d 0.98)×(sin² 1.00); df 1.02 = 1.63×0.68×0.92
(compensated by our smaller r_f). Mo pd 1.70 (k²/N 1.27, ratios 1.12, sin² 1.19). Zr pd 2.40 and Y pd 3.29 from the DOS ratios
r_p r_d (2.5×). sin²(δ_{l+1}−δ_l) agree within 10 % everywhere → the matrix-element identity is fine; the difference is in the
electronic-structure inputs: N(E_F) (our pure Nb 6.95/spin vs 9.71 — pure elements are not k-converged at bz20), E_F reference
(our k² used v(R⁻) = −0.109 Ry instead of the MT zero 0 → k² 0.788 vs their 0.676; η_B is invariant to this, the factor split is not)
and the partial-DOS ratios (Zr, Y, Ti: p and d). Caveat: the MT potential step ΔV at R_MT (v(R⁻) → 0) contributes ΔV·u_l(R)u_{l+1}(R)
to the rigid-MT matrix element (δ-function in dV/dr); our interior integral/surface form omits it (~5 % for Nb pd) — to be checked.

## Variant G = GG(7) with the true muffin-tin zero = surface form + MT-step term (11_gg_mtz0.py) — 2026-10-23
AkaiKKR's v3 is referenced to the MT zero (V = 0 outside R_MT); the last kept mesh value v(R⁻) ≠ 0 (Nb −0.109, Ta in alloy −0.154,
Ti in alloy **+0.097** Ry), i.e. the MT potential has a step ΔV = −v(R⁻) at R. In the rigid-MT model that step moves with the atom and
contributes ΔV·u_l(R)u_{l+1}(R) to M (δ-function in dV/dr). Verified to 4 decimals: GG(7) with free-wave matching at k² = E_F − 0
equals interior integral + step term, for every channel (Nb pd: −0.6535 + 0.0924 = −0.5611 = M_GG). Variant B (surface/interior)
omits it; it is NOT small: alloy Ta pd 120 → 76, Nb 99 → 76, Hf 187 → 116, Zr 172 → 120, **Ti 29 → 43** (ΔV < 0 raises it), df +15 %.
| | B (surface, MT) | **G (GG, MTZ=0)** | reference |
|---|---|---|---|
| pure Nb (bz20) | 12.8/89/93 = 195 | 8.5/66/103 = **177** | Papa 9.7/56/91 = 157 |
| pure Mo | 4.7/35/104 = 143 | 3.0/26/112 = 141 | Papa 1.4/21/97 = 119 |
| pure Ti | 3.4/63/69 = 136 | 2.3/47/75 = 124 | Papa 2.7/29/68 = 100 |
| alloy Ta | 4.5/120/111 = 236 | 1.7/76/129 = 207 | Jaśk. 0.9/51/102 = 154 |
| alloy Ti | 3.0/29/70 = 102 | 4.8/43/63 = 111 | Jaśk. 4.9/46/76 = 126 |
| alloy η_mix | 226 | **202** | 157 |
G is the physically complete rigid-MT (GG) value in the MT convention; the remaining 1.1–1.3× vs Papa/Kraków now looks like
N(E_F)/partial-DOS differences (pure Nb N(E_F) 28 % below Papa at bz20; bz30 test running). bz40 fails in AkaiKKR bzmesh.

### Baseline fit with variant G (`--cutoff-mode gg`; 06 … gg → eta_gg.csv; 07 compares A/B/G)
Set B, θ_D mix, μ* = 0.2 (n=74 / bcc 65): τ A 0.722/0.739, B 0.701/0.725, **G 0.675/0.705**; r 0.854 / 0.877 / 0.861;
MAE 1.95 / 2.09 / **1.39** K; bias +1.48 / +1.68 / **−0.64** K; median ratio 1.22 / 1.28 / 0.90; rms log 0.327 / 0.309 / 0.432
(G at its own best μ* = 0.17: rms log 0.297, τ 0.684). λ_G/λ_A = 0.77–1.03 (median 0.90), ρ(x_Ti) = −0.95: Ti-rich alloys drop
18–23 % (Ti50ZrNbHfTa50 7.2 → 2.5 K vs exp 6.0 — over-corrected), Mo/W/Hf-rich unchanged.
Verdict: G lowers the absolute scale (bias ≈ 0 at μ* 0.2) but orders slightly worse than A; with refitted μ* all three are
equivalent within noise. Production decision (keep A) unchanged.

### Pure Nb at Papaconstantopoulos' lattice constant (a = 6.20), bzqlty 30 — 2026-10-23 (bz40 exceeds AkaiKKR bzmesh)
`results/test.nrl.ew0.6/Nb_a6.20_{nrl_vbh,sra_pbe}_bz30`. NRL+LDA: E_F = 0.679 Ry (Papa 0.676), ratios n_l/n_l^(1) = 0.92/3.04/0.68/4.04
(Papa 0.81/2.93/0.68/3.85), sin²(δ_{l+1}−δ_l) = 0.29/1.00/0.83 (Papa 0.29/1.00/0.82) → every ingredient of GG(7) agrees within 10 %
**except the total DOS: N(E_F) = 7.84/Ry/spin vs 9.71** (bz20 SRA-PBE at our a0 gave 6.95; bz30 SRA-PBE 7.44). η_G = 14.2/72.4/120.6 = 207
vs Papa 9.7/56/91 = 157: the ratio 1.29–1.32 in pd/df is exactly the 1/N factor (9.71/7.84 = 1.24) plus r_s. With Papa's N our η_G
would be ≈ 167 (6 % high). Interpretation: AkaiKKR's site DOS (Σ_l dosef, l ≤ 3) is ~20 % below the APW total n(E_F), which includes
the interstitial region (and/or residual k-convergence: N rose 7 % from bz20 → bz30). For the alloy N(E_F) matched Jaśkiewicz within
5 % while η_G/η_Jaśk = 1.29; there the likely remaining factor is their MT-zero/k² convention (potential "extending to the WS sphere",
Rajput 1996) — not testable without their code.
**Final picture:** matrix elements (phase shifts), DOS ratios and the GG formula are all consistent with the literature; our η is
~1.25× the 1977/Kraków values through the 1/N(E_F) normalisation of the KKR site DOS. A constant factor → absorbed by μ*.

## Phase shifts δ_l — how they are computed (answer recorded 2026-10-23)
At R_MT the AkaiKKR function u_l = rR_l (from `rstr_real_ef`, any normalisation) is matched in value and derivative
(derivative from a CubicSpline in ln r, divided by R) to the free solution for the constant potential V_MTZ outside the sphere:
u_l(r) = A·r·j_l(kr) + B·r·y_l(kr), k = √(E_F − V_MTZ); 2×2 solve for A, B; δ_l = atan2(−B, A) (so R_l ∝ j_l cosδ_l − y_l sinδ_l).
Equivalent: tan δ_l = (k j_l′ − D_l j_l)/(k y_l′ − D_l y_l) with D_l the logarithmic derivative. δ_l is normalisation-independent;
sin²(δ_{l+1}−δ_l) is branch-independent (δ_p printed as 2.70 ≡ −0.44 + π). Check: for NRL functions M_free = −sin(δ_{l+1}−δ_l) to 1e-4.
**Scripts 09/10 used k² = E_F − v(R⁻)** (potential just inside the sphere, −0.109 Ry for Nb) → consistent with the interior integral
(variant B) but not the GG phase shifts; **script 11 and the pure-Nb bz30 table use the true MT zero, k² = E_F** → variant G
(= interior integral + MT-step term). The last pure-Nb table lists, per row, every ingredient of GG(7): k², n_l/n_l^(1), sin²Δδ,
N(E_F)/spin and η_G; the Papaconstantopoulos row is the expected value, and only N(E_F) disagrees (7.84 vs 9.71/spin).

## Compact comparison with Jaśkiewicz 2016 (alloy Ta34Nb33Hf8Zr14Ti11; totals only, no channel split) — 2026-10-23
N(E_F) per element both spins (1/Ry), η per element (mRy/a₀²). Ours: SRA, PBE, bz20, a0 3.365 Å. λ, Tc with θ_mix 276 K, ⅗θ², C_THEORETICAL.

| | N(E_F) Jaśk. | N(E_F) ours | η Jaśk. | η A (cutoff, prod.) | η B (surface/MT) | η G (GG, MTZ=0) |
|---|---|---|---|---|---|---|
| Ta | 17.2 | 16.4 | 154 | 232 | 236 | 207 |
| Nb | 18.8 | 18.0 | 162 | 239 | 218 | 202 |
| Hf | 16.2 | 15.4 | 161 | 161 | 286 | 231 |
| Zr | 17.2 | 16.9 | 176 | 185 | 288 | 246 |
| Ti | 26.6 | 25.3 | 126 | 187 | 102 | 111 |
| **η_mix** | | | **157** | **217** | **226** | **202** |
| ratio to Jaśk. | | 0.95–0.98 | 1 | 1.38 | 1.44 | 1.29 |
| λ (our conventions) | | | 0.76 | 1.04 | 1.09 | 0.97 |
| Tc μ*=0.2 [K] | | | 3.7 | 9.7 | 10.7 | 8.2 |
| Tc μ*=0.13 [K] | | | 7.5 | 14.7 | 15.7 | 13.0 |
| Jaśkiewicz own (λ 1.16, θ 243 K, ½θ²) | | | | Tc 10.7 (μ* 0.2) / 15.2 (μ* 0.13) | | exp 7.3 K |

Reading: DOS agrees within 5 %; η_mix is 1.3–1.45× theirs in every variant; per element the three variants disagree most for
Hf/Zr (G, B high) and Ti (A high); G is the closest overall (ratio 1.29 ≈ the same 1.25–1.3 seen vs Papaconstantopoulos for pure Nb).

## Bootstrap A vs B vs G (2026-10-23, set B n=74, θ_mix, μ*=0.2; 4000 paired resamples of fit_cutoff_vs_surface_rows.csv)
τ: A 0.722 [0.65, 0.79], B 0.701 [0.62, 0.77], G 0.675 [0.59, 0.75]. Paired differences: A−G +0.047 [−0.011, +0.111] (P(≤0) 0.07),
A−B +0.021 [−0.030, +0.073], B−G +0.026 [+0.002, +0.055]. Best μ* (rms log): A 0.205, B 0.21, G 0.17.
Median Tc_KKR/Tc_exp for Ti ≥ 30 at.% (n=10) vs rest: A 1.16 / 1.23 (uniform), B 0.95 / 1.29, **G 0.55 / 0.94** → G introduces a
Ti-dependent bias (Ti-rich alloys 2× too low) that A does not have; A's uniformity may be a compensation (cutoff inflates Ti pd).
