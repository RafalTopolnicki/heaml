# Artificially Large λ from Forcing BCC: Two Failure Patterns

Both patterns described here produce λ >> 1 and correspondingly inflated Tc predictions,
but for different physical reasons. In both cases the root problem is the same: the KKR
calculation assumes a BCC virtual crystal, and the composition is physically unsuited to it.
The BCC assumption distorts the elastic response, collapses θ_D, and through λ ∝ 1/θ_D²
drives λ to unphysical values.

---

## The λ formula

$$\lambda = \frac{\sum_i c_i \, \eta_i}{M_\text{mix} \cdot \theta_D^2} \cdot C$$

$C$ is a theoretical constant from the Debye model. The key point is the **quadratic
dependence on θ_D**: halving the Debye temperature quadruples λ. Both patterns below
collapse θ_D — by different mechanisms.

---

## Pattern 1 — BCC tetragonal instability from low VEC (La/Y/Sc-rich alloys)

**Example:** `Ti0.0612 Nb0.0599 Zr0.3550 Hf0.1002 Ta0.0262 Sc0.1778 Mo0.0092 W0.0031 Y0.0816 La0.1259`
→ λ = 4.06, λ_nocutoff = 126

### The η numerator is not the problem

| Element | c_i  | η_total | c_i × η_i |
|---------|------|---------|-----------|
| Zr      | 0.355 | 0.152  | 0.0538    |
| Sc      | 0.178 | 0.120  | 0.0214    |
| Hf      | 0.100 | 0.161  | 0.0161    |
| La      | 0.126 | 0.049  | 0.0061    |
| Y       | 0.082 | 0.095  | 0.0078    |
| Ti      | 0.061 | 0.163  | 0.0100    |
| Nb      | 0.060 | 0.196  | 0.0117    |
| others  | —     | —      | ~0.005    |
| **Sum** |       |        | **0.1357** |

Σ(c_i η_i) = 0.136 is typical for a 10-element refractory HEA. The numerator alone cannot explain λ > 4.

### θ_D collapses from BCC tetragonal instability

| Quantity          | Value          |
|-------------------|----------------|
| B₀                | 65.1 GPa       |
| **C' = (C11−C12)/2** | **−0.418 GPa** |
| C44 (scaled)      | 25.8 GPa       |
| Gv (Voigt)        | 15.3 GPa       |
| **Gr (Reuss)**    | **−1.07 GPa**  |
| Gh (Hill)         | 7.1 GPa        |
| vt                | 1027 m/s       |
| **θ_D (KKR)**     | **120.65 K**   |
| θ_D (mixture est.)| 271.7 K        |

**C' < 0** means C12 > C11: the BCC lattice has no resistance to tetragonal distortion.
Gr also goes negative. Gh is kept barely positive by C44, but at 7.1 GPa it is abnormally
low, giving vt ≈ 1027 m/s and θ_D ≈ 121 K — less than half the mixture estimate.
The (271/121)² ≈ **5× amplification** on λ follows directly.

| θ_D (K)             | λ    |
|----------------------|------|
| 271.7 (mixture est.) | 0.80 |
| 200                  | 1.48 |
| 150                  | 2.63 |
| **120.65 (KKR)**     | **4.06** |

### Why C' < 0: VEC too low to stabilize BCC

VEC is the concentration-weighted mean valence electron count per atom; here **mean VEC = 3.73**.

| Element    | VEC | BCC tendency                                        |
|------------|-----|-----------------------------------------------------|
| Mo         | 6   | strong BCC stabilizer                               |
| Ta, Nb     | 5   | strong BCC stabilizer                               |
| Ti, Zr, Hf | 4  | moderate BCC stabilizer                             |
| Sc, Y      | 3   | weak; HCP ground state                              |
| La         | 3   | very weak; FCC/DHCP ground state, nearly empty d-band |

La (12.6%) + Y (8.2%) + Sc (17.8%) = **~39% of the alloy**. These elements have nearly
empty d-bands and contribute little to the d–d bonding that provides BCC shear rigidity.
La is a lanthanide that naturally adopts FCC/DHCP; its virtual-crystal contribution to the
BCC potential is physically anomalous. Together they pull mean VEC to 3.73, well below
the ~4.5 threshold for robust BCC stability (Guo et al. criterion), and the BCC lattice
softens against tetragonal shear until C' < 0.

**Diagnostic:** C' < 0 and/or Gr < 0, mean VEC < ~4, large La/Y fractions.

---

## Pattern 2 — Soft BCC from extreme single-element dominance (Sc-heavy alloys)

**Example:** `Ti0.1179 Nb0.1848 Zr0.0383 Hf0.0208 Ta0.0205 Sc0.5983 Mo0.0193 W0.0000 Y0.0000 La0.0000`
→ λ = 4.54, λ_nocutoff = 29.7

Here **C' = +2.03 GPa** — the BCC structure is not tetragonally unstable. Yet λ > 4.

### Sc at 60% dominates the numerator

| Element | c_i   | η_total | c_i × η_i | % of sum |
|---------|-------|---------|-----------|---------|
| **Sc** | **0.598** | 0.131 | **0.0783** | **49.5%** |
| Nb     | 0.185 | 0.209   | 0.0387    | 24.5%   |
| Ti     | 0.118 | 0.187   | 0.0221    | 14.0%   |
| others | —     | —       | ~0.018    | 11.9%   |
| **Sum** |      |         | **0.1583** |        |

Σ(c_i η_i) = 0.158 is elevated, but Sc's η = 0.131 is not anomalous on its own. The high
contribution is purely a concentration effect. Without Sc in the numerator, λ would be **2.3**.

### θ_D is still collapsed — now from Sc's intrinsically soft BCC

| Quantity          | Value         |
|-------------------|---------------|
| B₀                | 78.3 GPa      |
| **C' = (C11−C12)/2** | **+2.03 GPa** |
| C44 (scaled)      | 19.4 GPa      |
| Gv (Voigt)        | 12.5 GPa      |
| Gr (Reuss)        | 4.4 GPa       |
| **Gh (Hill)**     | **8.4 GPa**   |
| vt                | 1307 m/s      |
| **θ_D (KKR)**     | **159 K**     |
| θ_D (mixture est.)| 341 K         |

Sc's ground state is HCP. Forced into BCC at 60% concentration, it makes the alloy
behave like a Sc alloy — with Sc's naturally low BCC shear modulus. C' is positive
but small (2 GPa), C44 is modest, and Gh = 8.4 GPa is again abnormally low.
The (341/159)² ≈ **4.6× amplification** on λ follows.

| θ_D (K)             | λ    |
|----------------------|------|
| 341 (mixture est.)   | 0.99 |
| 250                  | 1.85 |
| 200                  | 2.89 |
| **159 (KKR)**        | **4.54** |

**Diagnostic:** C' positive but small, one element > ~50% with HCP/FCC ground state,
θ_D well below mixture estimate, λ_nocutoff only modestly larger than λ (no runaway
sp-channel divergence, unlike Pattern 1).

---

## Shared root cause

Both patterns share the same underlying logic:

1. KKR assumes a BCC virtual crystal regardless of whether the composition would
   physically adopt BCC.
2. Elements foreign to BCC (La, Y with nearly empty d-bands; Sc at extreme concentration)
   make the BCC elastic response anomalously soft.
3. θ_D collapses — either catastrophically (C' < 0, Pattern 1) or severely (soft BCC Sc, Pattern 2).
4. λ ∝ 1/θ_D² amplifies the result by 4–5×, producing λ >> 1 and inflated Tc.

The effect is not real electron–phonon coupling — it is the BCC structural assumption
failing for compositions that would not adopt BCC in practice.

## Screening criteria

Flag a result as potentially artifactual when any of the following appear:

| Criterion | Pattern |
|-----------|---------|
| C' < 0 | 1 |
| Gr < 0 | 1 |
| Mean VEC < 4.0 | 1 |
| La + Y fraction > ~15% | 1 |
| Single element > ~50% with HCP/FCC ground state | 2 |
| θ_D (KKR) < 0.65 × θ_D (mixture) | both |
| λ_nocutoff >> λ (factor > 10) | 1 (runaway sp channel) |
