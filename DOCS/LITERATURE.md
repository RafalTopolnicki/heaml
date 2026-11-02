# Spis literatury — prace istotne dla projektu (z uzasadnieniem)

Założony 2026-11-02. Każda pozycja ma: pełny opis, DOI lub URL, **co z niej bierzemy** i **stopień weryfikacji**:
- `PEŁNY`: przeczytany pełny tekst;
- `ABS`: tylko streszczenie lub fragmenty z wyszukiwarki;
- `SEC`: wartość z pracy cytującej lub z przeglądu.

Sekcje 1–2 zawierają prace potrzebne do interpretacji i poprawek modelu. Sekcja 3 zawiera spis wszystkich prac, z których
braliśmy eksperymentalne Tc: dane bibliograficzne, stop i raportowane Tc. Same dane liczbowe trzymamy w `results/Tc_literature.xlsx`
i `results/Tc_literature_V_additions.csv`.

---

## 1. Fluktuacje spinowe w V, Nb i Ti–V — uzasadnienie poprawki λ_es (`Tc_corr`)

Kontekst (2026-11-02): w optymalizacjach składu B (TiNbZrHfTaMoWYVSc) najlepszym stopem dwuskładnikowym wychodzi V–Ti
(Tc_mu0.1 ≈ 20 K, Tc_corr ≈ 18.5 K). Eksperyment daje 6–7 K, a w literaturze kolejność jest Nb–Zr > Nb–Ti > V–Ti.
Zawyżenie V–Ti wynosi ×2.5–2.8, podczas gdy dla Nb–Zr ×1.6–1.7, a dla Nb–Ti ×1.4.
Notatka: `DOCS/ROZMOWA_2026-11-02_optB_VTi.md`.

### 1.1 Matin i in., EPJ B (2014) — Ti–V: Tc obniżone przez lokalne fluktuacje spinowe
- M. Matin, L. S. Sharath Chandra, M. K. Chattopadhyay, S. B. Roy i in., *Eur. Phys. J. B* (2014),
  doi [10.1140/epjb/e2014-50036-2](https://doi.org/10.1140/epjb/e2014-50036-2).
  Omówienie: [EPJ B Highlight](https://www.epj.org/epjb-news/811-ti-v-alloy-superconductivity-inherent-not-accidental).
- **Co bierzemy:**
  - Tc stopów TiₓV₁₋ₓ (x = 0.4–0.8) jest wynikiem konkurencji fluktuacji spinowych i sprzężenia elektron–fonon;
  - oszacowanie BCS/McMillan daje ok. **20 K**, a zmierzone jest poniżej 10 K.
- **Znaczenie dla nas:** nasze 17–20 K dla V–Ti to „poprawny McMillan bez paramagnonów”. Brak jest fizyczny, a nie
  numeryczny, i uzasadnia poprawkę elektron–paramagnon. Potrzebne λ_es ≈ 0.25–0.27, a obecne `Tc_corr` daje 0.08.
- **Weryfikacja:** `ABS` (highlight EPJ i fragmenty). Pełnego tekstu nie czytaliśmy; liczby trzeba sprawdzić w oryginale.

### 1.2 μSR na Ti55V45 — bezpośrednia obserwacja fluktuacji spinowych (Warwick, 2015)
- J. Barker i in., prezentacja konferencyjna (Warwick, Superconductivity & Magnetism group), Ti55V45, Tc = 7.1 K:
  [fusm2015.pdf](https://text.www2.warwick.ac.uk/fac/sci/physics/research/condensedmatt/supermag/whoswho/joel_barker/fusm2015.pdf).
- **Co bierzemy:**
  - fluktuacje spinowe pojawiają się poniżej ok. 126 K;
  - głębokość wnikania pasuje do izotropowego modelu s-wave w granicy silnego sprzężenia;
  - Tc stopów Ti–V jest wyraźnie niższe niż przewiduje formalizm McMillana.
- **Znaczenie dla nas:** niezależne, mikroskopowe potwierdzenie obrazu z 1.1.
- **Weryfikacja:** `ABS` (slajdy, fragmenty). To nie jest praca recenzowana; nie cytować jako jedynego źródła.

### 1.3 Ujemny TCR od wędrownych fluktuacji spinowych w V₀.₃Ti₀.₇ (arXiv 2025)
- arXiv [2503.12245](https://arxiv.org/html/2503.12245v1).
- **Co bierzemy:**
  - fluktuacje spinowe występują także po stronie bogatej w Ti (V30 Ti70);
  - klastry Gd tłumią fluktuacje i zwężają przejście nadprzewodzące.
- **Znaczenie dla nas:** poprawka λ_es musi działać w całym zakresie V–Ti, nie tylko dla stopów bogatych w V.
  Potrzebne λ_es jest prawie stałe: 0.25 dla Ti30V70 i 0.27 dla Ti70V30.
- **Weryfikacja:** `ABS`, preprint.

### 1.4 Rietschel i Winter, PRL 43, 1256 (1979) — fluktuacje spinowe w Nb i V
- H. Rietschel, H. Winter, *Phys. Rev. Lett.* **43**, 1256 (1979),
  doi [10.1103/PhysRevLett.43.1256](https://dx.doi.org/10.1103/PhysRevLett.43.1256).
- **Co bierzemy:**
  - reanaliza ciepła właściwego i podatności dla Nb i V oraz rozwiązanie równań Eliashberga z kanałem particle–hole;
  - fluktuacje spinowe **wyraźnie obniżają Tc**, a ich wkład do wzmocnienia masy jest rzędu **0.2**;
  - dla Nb cytowane jest λ_spin ≈ 0.2 (`SEC`).
- **Znaczenie dla nas:** klasyczne źródło skali λ_sf ≈ 0.2–0.3 dla V i Nb. Zgadza się z λ_es ≈ 0.25 potrzebnym
  dla V–Ti; obecne α = 0.3 daje za małe wartości.
- **Weryfikacja:** `ABS`. Autorstwo przypisane na podstawie cytowania w pracy 1.5. Liczby λ_sf dla V odczytać z pełnego tekstu.

### 1.5 Rietschel i in., PRB (1981) — fluktuacje spinowe a termodynamika nadprzewodzącego V i VN
- „Importance of spin fluctuations for the thermodynamic properties of superconducting V and VN”, *Phys. Rev. B* (1981):
  [rekord](https://ouci.dntb.gov.ua/works/2loOENpl).
- **Co bierzemy:** rozszerzenie analizy z 1.4 na V, w tym wartości λ_sf i μ* dla V do kalibracji.
- **Znaczenie dla nas:** docelowe źródło liczbowej wartości λ_sf(V) do kalibracji `Tc_corr`; czysty V ma
  eksperymentalne Tc 5.4 K, a my dajemy ok. 11 K.
- **Weryfikacja:** tylko rekord bibliograficzny. **Do przeczytania.**

### 1.6 Ab initio modelowanie stopów nadprzewodzących (arXiv 2024)
- arXiv [2406.15174](https://arxiv.org/pdf/2406.15174).
- **Co bierzemy:**
  - μ* dopasowane z obliczeń z fluktuacjami spinowymi wynosi **0.419 dla V** i **0.277 dla Nb**;
  - konwencjonalne μ* ≈ 0.1–0.13 nie stosuje się do V i Nb z powodu silnych fluktuacji ferromagnetycznych.
- **Znaczenie dla nas:**
  - alternatywna parametryzacja poprawki: efektywne μ* zależne od pierwiastka zamiast λ_es;
  - potwierdza, że V wymaga silniejszej poprawki niż Nb.
- **Weryfikacja:** `ABS` (fragmenty).

### 1.7 Bose, J. Phys.: Condens. Matter 21, 025602 (2009) — formuły λ_es (już używane w kodzie)
- S. K. Bose, *J. Phys.: Condens. Matter* **21**, 025602 (2009).
- **Co bierzemy:** równania 26–27, czyli λ_es = α Σ xᵢ sᵢ ln(1/(1−sᵢ)) oraz zmodyfikowana formuła McMillana z λ_es.
  Zaimplementowane w `process_kkr.compute_tc_corrected`.
- **Znaczenie dla nas:** podstawa obecnej poprawki 2 w `Tc_corr`.

---

## 2. Stopy HEA i MEA bogate w V — dane eksperymentalne (kandydaci do bazy walidacyjnej)

Wartości w `results/Tc_literature_V_additions.csv` (20 stopów HEA i 17 stopów dwuskładnikowych) zebraliśmy wcześniej.
Poniżej są **nowe** pozycje z wyszukiwania 2026-11-02; żadna nie jest jeszcze w CSV.

| stop | Tc [K] | źródło | uwagi | weryfikacja |
|---|---|---|---|---|
| (VNb)₀.₇₅₃(TiHf)₀.₂₄₇ | 5.2 | [Materials 18, 2747 (2025), MDPI](https://www.mdpi.com/1996-1944/18/12/2747) | bcc; wariant z Tb ma 4.6 K i jest wielofazowy | `ABS` |
| ScTiVNbCu (równoatomowy) | 6.75 (Cp); ρ ok. 7.3 | [arXiv 2501.01887](https://arxiv.org/pdf/2501.01887) | bcc, typ II; zawiera Cu, które nie należy do naszego zestawu pierwiastków | `ABS` |
| TiVZrNb | 7.9 (lany), 9 (wyżarzony 800 °C) | przegląd [High entropy superconductors, Phys. Rep. (2025)](https://www.sciencedirect.com/science/article/pii/S0370157325002352) | eutektyka bcc+hcp, nie jest jednofazowy | `SEC` |
| Ta₀.₂Nb₀.₂V₀.₂Ti₀.₂Zr₀.₂ | 6.19 (onset) | [arXiv 2504.08124](https://arxiv.org/html/2504.08124v1) = PRMater 9, 064801 (2025) | bcc; w CSV mamy 5.13 K (Cp) z drugiej ręki | `ABS` |
| Ta₀.₂Nb₀.₂V₀.₂Ti₀.₂Hf₀.₂ | ok. 5 (onset) | jak wyżej | bcc; w CSV mamy 4.0 K (Cp) | `ABS` |
| Hf21Nb25Ti15V15Zr24 | 5.3 | [arXiv 1904.03291](https://ar5iv.labs.arxiv.org/html/1904.03291) (Ishizu 2019) | już w CSV | `ABS` |
| V–Nb–Mo–Al–Ga | — | [Sci. China Mater. (2019)](https://link.springer.com/article/10.1007/s40843-019-1237-5) | po wyżarzaniu przechodzi w fazę A15, więc nie nadaje się do bazy bcc | `ABS` |
| V, V–Ti pod ciśnieniem | — | [JETP Lett. (2009)](https://link.springer.com/article/10.1134/S0021364009050087) | Tc(p) dla V i V–Ti; możliwy test zależności od objętości | `ABS` |

**Stopy dwuskładnikowe z V już w CSV** (źródła pierwotne nieczytane, wartości `SEC` przez Senga 2026, tabela 2):
- Ti–V, Cheng i in., *Phys. Rev.* 126, 2030 (1962): 6.1–7.3 K, z θ_D i γ;
- V–Nb, V–Ta, Corsan, *phys. stat. sol.* 40, 657 (1970);
- V–Mo, Kuentzler, *Solid State Commun.* 59, 299 (1986).

Te stopy mają posłużyć do kalibracji λ_es; zob. plan w notatce z rozmowy.

---

## 3. Prace, z których braliśmy dane eksperymentalne Tc (stan 2026-11-02)

Przejrzane źródła w repozytorium:
- `results/Tc_literature.xlsx`: 124 wiersze, 48 prac; jedyne źródło prawdy dla bazy referencyjnej;
- `results/Tc_baseline.xlsx`: stara baza; wszystkie stopy przeniesione do `Tc_literature.xlsx`, a czyste pierwiastki są w 3.4;
- `results/Tc_literature_V_additions.csv`: stopy z V, w tym dwuskładnikowe;
- `papers_data/bcc_hea_superconducting_tc_summary.xls`: zestawienie, z którego brały się wpisy w `0_compare_with_known.sh`;
- skrypty startowe:
  - `bash.old/0_compare_with_known.sh`, `0_compare_with_known_bz10-10-20.sh`, `0_compare_with_known_V.sh`;
  - `bash.old/3_run_validation_binaries*.sh`, `3_run_validation_NbTiZrHf.sh`, `3_run_validation_TiNb_ew_scan.sh`;
  - `slurm/submit_compare.sh`;
- `results/literature_baseline_extension.md` i `VALIDATION_NOTE.md`.

Wszystkie prace cytowane w komentarzach skryptów `0_compare_with_known*.sh` są w `Tc_literature.xlsx`. Skrypty
walidacji stopów dwuskładnikowych nie podają wartości eksperymentalnych; ich odniesienia są w 3.5.

**Uwaga o pracach teoretycznych jako źródle.** Kilka wierszy z dawnego `Tc_baseline.xlsx` ma w kolumnie Link pracę
obliczeniową, która tylko cytuje eksperyment:
- Ta34Nb33Hf8Zr14Ti11 (7.3 K) jest przypisany do Jaśkiewicz 2016, a pomiar pochodzi z Koželj i in., PRL 113, 107001 (2014);
- TaNb67HfZrTi33 (7.7 K) jest przypisany do Jaśkiewicz 2019, a pomiar pochodzi z von Rohr 2016;
- podobnie mogą wyglądać wiersze Idczak (PRB 112, 014513) i Sobota (PRB 106, 184512).

Wartości są oznaczone „primary paper not re-checked”.

### 3.1 Prace w `results/Tc_literature.xlsx` (baza referencyjna, źródło prawdy)

Kolumny: Tc = wartość główna (kolumna R arkusza); metoda; Tc alt = inne definicje; źródło wartości (TAB/FIG-VEC/FIG-LBL/FIG-EST/SEC/ABS, legenda arkusza); status; pochodzenie w projekcie.

**1. Chakrabarty i in., *Journal of Applied Physics* 137, 215901 (2025)** — [10.1063/5.0265943](https://doi.org/10.1063/5.0265943)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbTa55HfTiZr45 | 7.2 | ρ, ambient pressure |  | SEC (conference abstract / summary xls BULK-009) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |

**2. Gao i in., APL 120,092602 (2022)** — [10.1063/5.0083365](https://doi.org/10.1063/5.0083365)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNb70HfZrTi50_Gao | 7.13 | as-cast |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**3. Hattori i in., J.Alloys Metall.Syst. 3,100020 (2023)** — [10.1016/j.jalmes.2023.100020](https://doi.org/10.1016/j.jalmes.2023.100020)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti35Hf25Nb25Ta5Re10 | 3.95 | Cp |  | TAB (verified, arXiv 2307.01958) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| Ti30Hf20Nb35Ta5Re10 | 4.38 | Cp |  | TAB (verified, arXiv 2307.01958) | caution – multiphase | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| Ti25Hf15Nb35Ta15Re10 | 4.1 | Cp |  | TAB (verified, arXiv 2307.01958) | caution – multiphase | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| Ti20Hf10Nb35Ta25Re10 | 3.62 | Cp |  | TAB (verified, arXiv 2307.01958) | caution – multiphase | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| Ti15Hf5Nb35Ta35Re10 | 3.25 | Cp |  | TAB (verified, arXiv 2307.01958) | caution – multiphase | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |

**4. Hidayati i in., ActaMater 261,119420 (2023)** — [10.1016/j.actamat.2023.119420](https://doi.org/10.1016/j.actamat.2023.119420)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb40Hf20Zr20Ti20 | 7.99 | Cp |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**5. Idczak i in., PRB 112,014513** — [10.1103/fnz7-bzmw](https://doi.org/10.1103/fnz7-bzmw)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbHf75TiZr25 | 7.15 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 5 |
| NbZr75TiHf25 | 8.17 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 6 |
| NbTi75ZrHf25 | 7.76 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 7 |

**6. Idczak i in., *Materials* 16, 5814 (2023)** — [10.3390/ma16175814](https://doi.org/10.3390/ma16175814)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb34Ti33Zr14Ta11Hf8 | 7.5 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 8 |

**7. Idczak i in., PRB 114,094507 (2026)** — [10.1103/k1yr-t59z](https://doi.org/10.1103/k1yr-t59z)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| YNbTiZrHf | 6.1 | ρ, χ, ac-χ, Cp (≈6.1) |  | ABS | caution – minor second phase | lit search 2026-10-19 (literature_baseline_extension.md) |

**8. Ishizu i in., Results Phys. 13, 102275 (2019)** — [10.1016/j.rinp.2019.102275](https://doi.org/10.1016/j.rinp.2019.102275)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Hf21Nb25Ti15V15Zr24 | 5.3 | chi_ac (intercept) |  | TAB | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**9. Jangid i in., Appl. Phys. Lett. 124, 192602 (2024)** — [10.1063/5.0200805](https://doi.org/10.1063/5.0200805)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| ScVTiHfNb | 4.17 | chi (1 mT) | 4.47 (rho mid) / 3.72 (Cp) | TAB | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**10. Jaskiewicz i in., Phys.Stat.Sol.RRL 10,415** — [10.1002/pssr.201600056](https://doi.org/10.1002/pssr.201600056)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ta34Nb33Hf8Zr14Ti11 | 7.3 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 4 |

**11. Jaskiewicz i in., PRB 100,184503 (2019)** — [10.1103/PhysRevB.100.184503](https://doi.org/10.1103/PhysRevB.100.184503)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNb67HfZrTi33 | 7.7 | as reported (bulk) | 7.81 (χ, von Rohr 2016 Fig. 4, FIG-VEC); 7.70 (Cp, von Rohr 2016 Table II) | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 3 |

**12. Kim i in., *Acta Materialia* 186, 250-256 (2020)** — [10.1016/j.actamat.2020.01.007](https://doi.org/10.1016/j.actamat.2020.01.007)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNb2HfZrTi | 7.9 | ρ onset | 7.8 (ρ zero resistance) | TAB per summary xls (BULK-003) | baseline | Tc_baseline.xlsx row 28 |

**13. Kitagawa i in., *Journal of Alloys and Compounds* 924, 166473 (2022)** — [10.1016/j.jallcom.2022.166473](https://doi.org/10.1016/j.jallcom.2022.166473)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| HfMoNbTiZr | 4.1 |  |  | TAB per bcc_hea_superconducting_tc_summary.xls (BULK-004) | baseline | Tc_baseline.xlsx row 26 |

**14. Kitagawa i in., Mater.Today Commun. 40,109690 (2024)** — [10.1016/j.mtcomm.2024.109690](https://doi.org/10.1016/j.mtcomm.2024.109690)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbScTiZr_Kitagawa | 8.11 | Cp, as-cast | 8.71 / 9.36 / 9.13 / 9.03 (Cp, annealed 400/600/800/1000 °C) | TAB (verified) | caution – multiphase eutectic | lit search 2026-10-19 (literature_baseline_extension.md) |

**15. Košuth i in., Low Temp.Phys. 50,663 (2024)** — [10.1063/10.0027922](https://doi.org/10.1063/10.0027922)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TiZrNbHf_Kosuth | 6.3 | χ (~6.3) |  | ABS | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TiZrNb_Kosuth | 8.2 | χ (~8.2) |  | ABS | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**16. Krishnan i in., JALCOM 1036,181813 (2025)** — [10.1016/j.jallcom.2025.181813](https://doi.org/10.1016/j.jallcom.2025.181813)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNbZrTi_Krishnan | 8.12 | χ and ρ |  | ABS | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNbHfZr_Krishnan | 7.64 | χ and ρ | 6.50 (Cp, SEC) | ABS (Tc); SEC (Cp data) | caution – θD suspicious | lit search 2026-10-19 (literature_baseline_extension.md) |

**17. Krishnan i in., SuST 38,095009 (2025)** — [10.1088/1361-6668/ae01e7](https://doi.org/10.1088/1361-6668/ae01e7)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Hf35Nb10Ta35Mo10W10 | 4.06 | Cp |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| Hf35Nb22Ta23Mo10W10 | 4.82 | Cp |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| Hf35Nb35Ta10Mo10W10 | 5.91 | Cp | 6.05 (abstract max, probably ρ) | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**18. Krnel i in., Materials 15,1122 (2022) Krnel** — [10.3390/ma15031122](https://doi.org/10.3390/ma15031122)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| ScHfNbTaTi | 6.6 | ρ | 6.1 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 12 |
| ScHfNbTaTiZr | 7.2 | ρ | 6.6 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 13 |
| ScHfNbTaZr_bccL | 7.4 | ρ | 7.2 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 14 |
| ScHfNbTaZr_bccS | 7.4 | ρ | 7.2 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 15 |
| ScHfNbTi | 6.6 | ρ | 6.2 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 16 |
| ScHfNbZr | 6.8 | ρ | ~6 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 17 |
| ScHfTaTi | 6.4 | ρ | 6.0 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 18 |
| ScNbTaTiZr | 7.9 | ρ | 7.2 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 19 |
| ScNbTiZr | 8.4 | ρ | 7.3 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 20 |
| ScTaTiZr | 7.4 | ρ | 6.8 (Cp) | TAB (Tables 3–4, Europe PMC PMC8838456); a0 = BCC-phase a from Table 3 | caution – multiphase; corrected | Tc_baseline.xlsx row 21 |
| TaNbHfScTi | 6.6 | ρ | 6.1 (Cp) | TAB (Table 4) | caution – multiphase | Tc_baseline.xlsx row 35 |
| TaNbHfZrSc | 7.4 | ρ | 7.2 (Cp) | TAB (Table 4) | caution – multiphase; corrected | Tc_baseline.xlsx row 36 |
| TaNbScZrTi | 7.9 | ρ | 7.2 (Cp) | TAB (Table 4) | caution – multiphase | Tc_baseline.xlsx row 37 |
| TaNbHfZrTiSc | 7.2 | ρ | 6.6 (Cp) | TAB (Table 4) | caution – multiphase | Tc_baseline.xlsx row 38 |
| ScHfNbTi_nominal | 6.6 | ρ | 6.2 (Cp) | TAB (Table 4) | caution – multiphase | lit search 2026-10-19 (literature_baseline_extension.md) |
| ScHfNbZr_nominal | 6.8 | ρ | ~6 (Cp) | TAB (Table 4) | caution – multiphase | lit search 2026-10-19 (literature_baseline_extension.md) |
| ScHfTaTi_nominal | 6.4 | ρ | 6.0 (Cp) | TAB (Table 4) | caution – multiphase | lit search 2026-10-19 (literature_baseline_extension.md) |
| ScNbTiZr_nominal | 8.4 | ρ | 7.3 (Cp) | TAB (Table 4) | caution – multiphase | lit search 2026-10-19 (literature_baseline_extension.md) |
| ScTaTiZr_nominal | 7.4 | ρ | 6.8 (Cp) | TAB (Table 4) | caution – multiphase | lit search 2026-10-19 (literature_baseline_extension.md) |

**19. Kubo i in., JALCOM 1044,184531 (2025)** — [10.1016/j.jallcom.2025.184531](https://doi.org/10.1016/j.jallcom.2025.184531)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Hf10Nb25Sc25Ti20Zr20 | 6.83 | Cp, as-cast | 7.15 / 8.36 / 8.75 / 8.37 (annealed, up to 800 °C) | TAB (verified) | caution – multiphase eutectic | lit search 2026-10-19 (literature_baseline_extension.md) |
| Hf5Nb45Sc20Ti15Zr15 | 7.69 | Cp, as-cast | 7.64 / 8.17 / 9.28 / 9.39 (annealed) | TAB (verified) | caution – multiphase eutectic | lit search 2026-10-19 (literature_baseline_extension.md) |
| Hf5Nb45Sc10Ti5Zr35 | 7.73 | Cp, as-cast | up to 9.93 after 800 °C anneal | TAB (verified) | caution – multiphase eutectic | lit search 2026-10-19 (literature_baseline_extension.md) |

**20. Li i in., J. Phys. Chem. C 127, 16211 (2023)** — [10.1021/acs.jpcc.3c04125](https://doi.org/10.1021/acs.jpcc.3c04125)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TiVNbTa | 4.65 | rho/chi (bulk) | 4.38 (Cp) | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**21. Li i in., Mater. Today Commun. 38, 108444 (2024)** — [10.1016/j.mtcomm.2024.108444](https://doi.org/10.1016/j.mtcomm.2024.108444)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| V57Ti29W14 | 5 | rho/chi/Cp (abstract: roughly 5.0 K) | 5.57 (onset in text) | TAB | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**22. Li i in., Chin.Phys.Lett. 42,017401 (2025)** — [10.1088/0256-307X/42/1/017401](https://doi.org/10.1088/0256-307X/42/1/017401)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb50Ti25W25 | 4.2 | Cp | 4.86 (ρ, χ) | TAB (verified, arXiv 2412.08361) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| Nb50Ti25Mo25 | 2.8 | Cp | 3.22 (ρ, χ) | TAB (verified, arXiv 2412.08361) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**23. Li i in., SuST 38,015025 (2025)** — [10.1088/1361-6668/ad9fa8](https://doi.org/10.1088/1361-6668/ad9fa8)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| (TiHfTa)1-xNbx_x0 | 5.31 | ρ (50%) | 4.88 (Cp, Senga 2026) | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.035 | 5.61 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.125 | 5.82 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.33 | 7.06 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.44 | 7.41 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.57 | 8.19 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.75 | 8.82 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| (TiHfTa)1-xNbx_x0.9 | 9.11 | ρ (50%) |  | TAB (verified, arXiv 2501.02455) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**24. Maran i in., arXiv 2607.23056 (2026)** — arXiv:2607.23056

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb57Re10HfZrTi33 | 5.42 | ρ |  | TAB (preprint) | candidate – preprint | lit search 2026-10-19 (literature_baseline_extension.md) |
| Nb47Re20HfZrTi33 | 5.42 | ρ |  | TAB (preprint) | candidate – preprint | lit search 2026-10-19 (literature_baseline_extension.md) |

**25. Marik i in., *Journal of Alloys and Compounds* 769, 1059-1063 (2018)** — [10.1016/j.jallcom.2018.08.039](https://doi.org/10.1016/j.jallcom.2018.08.039)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbReZrHfTi | 5.3 | ρ, χ, Cp |  | TAB per summary xls (BULK-002) | baseline | Tc_baseline.xlsx row 27 |

**26. Motla i in., *Physical Review B* 105, 144501 (2022)** — [10.1103/PhysRevB.105.144501](https://doi.org/10.1103/PhysRevB.105.144501)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb60Re10Zr10Hf10Ti10 | 5.7 | χ, ρ, Cp |  | TAB (verified, arXiv 2112.02388) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |

**27. Nowak i in., Metall. Mater. Trans. A 55, 3789 (2024)** — [10.1007/s11661-024-07488-4](https://doi.org/10.1007/s11661-024-07488-4)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbTi67MoHfV33 | 4.8 | Cp/chi/rho |  | TAB | caution – minor second phase | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**28. Sang i in., SuST 38,055022 (2025)** — [10.1088/1361-6668/add097](https://doi.org/10.1088/1361-6668/add097)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| HfNbZrTi_crystal | 6 | Cp |  | SEC / ABS | caution – θD/γ suspicious | lit search 2026-10-19 (literature_baseline_extension.md) |

**29. Sarkar i in., Intermetallics 144, 107503 (2022)** — [10.1016/j.intermet.2022.107503](https://doi.org/10.1016/j.intermet.2022.107503)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| HfNbTaTiV_Sarkar | 5 | rho/chi |  | ABS | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**30. Seki i in., JSNM 37,1059 (2024)** — [10.1007/s10948-023-06643-z](https://doi.org/10.1007/s10948-023-06643-z)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbScTiZr_Seki | 7.9 | as-cast | 8.4 / 9.0 / 8.7 (annealed 400/800/1000 °C) | TAB (verified) | caution – multiphase eutectic | lit search 2026-10-19 (literature_baseline_extension.md) |

**31. Senga i in., Supercond. Sci. Technol. 39 (2026), arXiv 2602.19376** — [10.1088/1361-6668/ae37e4](https://doi.org/10.1088/1361-6668/ae37e4)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbTiZr | 7.98 | Cp |  | TAB (verified, arXiv 2602.19376) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| HfNbTi | 6.47 | Cp |  | TAB (verified, arXiv 2602.19376) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| HfNbZr | 7.07 | Cp |  | TAB (verified, arXiv 2602.19376) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| HfNbTa | 7.11 | Cp |  | TAB (verified, arXiv 2602.19376) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| HfNbTiVZr | 3.73 | Cp (midpoint) |  | TAB | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**32. Sharma i in., Mater. Today Commun. 44, 111968 (2025)** — [10.1016/j.mtcomm.2025.111968](https://doi.org/10.1016/j.mtcomm.2025.111968)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti25V25Nb40Ta10 | 5.2 | rho/chi | 4.50 (Cp) | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**33. Sharma i in., PRB 112,224515 (2025)** — [10.1103/8m3w-xc5x](https://doi.org/10.1103/8m3w-xc5x)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb25Ta25Ti25Zr25 | 7.75 | ρ midpoint | 8.0 (ρ onset) | TAB (verified, arXiv 2508.19584) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**34. Sharma i in., Phys. Rev. Mater. 9, 064801 (2025)** — [10.1103/PhysRevMaterials.9.064801](https://doi.org/10.1103/PhysRevMaterials.9.064801)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbTaTiVZr | 5.13 | Cp |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| HfNbTaTiV | 4 | Cp |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**35. Sobota i in., *Physical Review B* 106, 184512 (2022)** — [10.1103/PhysRevB.106.184512](https://doi.org/10.1103/PhysRevB.106.184512)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| NbTa67MoHfW33 | 4.3 |  |  | as in Tc_baseline.xlsx (primary paper not re-checked) | baseline | Tc_baseline.xlsx row 9 |

**36. Sobota i in., Acta Mater. 285, 120666 (2025)** — [10.1016/j.actamat.2024.120666](https://doi.org/10.1016/j.actamat.2024.120666)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti50ZrNbHfTa50 | 6 | Cp (~6.0) |  | SEC (PDF in papers_data/, not re-read) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| Ti50VNbHfTa50 | 5.9 | Cp |  | TAB | caution – compositional zones | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**37. Sobota i in., Appl. Phys. Lett. 129, 111905 (2026)** — [10.1063/5.0351820](https://doi.org/10.1063/5.0351820)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti47.5ZrNbTaHf47.5Sc5 | — | Tc not yet extracted |  | ABS | TODO – read paper | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**38. Sobota i in., Materials 19,2321 (2026) Sobota & Bartz** — [10.3390/ma19112321](https://doi.org/10.3390/ma19112321)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Nb67TiZrHf33_MA | 7 | onset ~7 (range 6–7.5) |  | TAB (verified, PMC) | caution – mechanically alloyed | lit search 2026-10-19 (literature_baseline_extension.md) |

**39. Uporov i in., Intermetallics 140,107394 (2022)** — [10.1016/j.intermet.2021.107394](https://doi.org/10.1016/j.intermet.2021.107394)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TiZrHfNb_Uporov | 6.3 | unknown |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**40. von Rohr i in., PNAS 113, E7144 (2016)** — [10.1073/pnas.1615926113](https://doi.org/10.1073/pnas.1615926113)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNb70ZrHfTi30 | 8.03 | ρ | 7.92 (χ ZFC 2 mT, FIG-VEC Fig. 4) | Tc ρ: TAB (Table I); Tc χ: FIG-VEC | baseline – corrected | Tc_baseline.xlsx row 30 |
| TaNb60ZrHfTi40 | 7.56 | ρ | 7.15 (χ ZFC 2 mT, FIG-VEC Fig. 4) | Tc ρ: TAB (Table I); Tc χ: FIG-VEC | baseline – corrected | Tc_baseline.xlsx row 31 |
| TaNb50ZrHfTi50 | 6.46 | ρ | 6.42 (χ ZFC 2 mT, FIG-VEC Fig. 4) | Tc ρ: TAB (Table I); Tc χ: FIG-VEC | baseline – corrected | Tc_baseline.xlsx row 32 |
| TaNb16ZrHfTi84 | 4.52 | ρ | 4.44 (χ ZFC 2 mT, FIG-VEC Fig. 4) | Tc ρ: TAB (Table I); Tc χ: FIG-VEC | baseline – corrected | Tc_baseline.xlsx row 33 |
| TaNb80ZrHfTi20 | 7.55 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb75ZrHfTi25 | 7.79 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb65ZrHfTi35 | 7.48 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb55ZrHfTi45 | 6.68 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb40ZrHfTi60 | 6.24 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb30ZrHfTi70 | 5.2 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb20ZrHfTi80 | 4.64 | χ ZFC 2 mT (slope ∩ normal state) |  | FIG-VEC (Fig. 4 marker coordinates; ±0.02 K) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| Nb67HfZrTi33 | 9.36 | χ ZFC 2 mT (tangent ∩ normal state) | 9.21 (M/M0=-0.5, FIG-VEC); label ≈9.2 (FIG-LBL) | FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb67Hf33 | 7.42 | χ ZFC 2 mT (tangent ∩ normal state) | 7.29 (M/M0=-0.5, FIG-VEC); label ≈7.3 (FIG-LBL) | FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNb67HfZr33 | 7.07 | χ ZFC 2 mT (tangent ∩ normal state) | 6.92 (M/M0=-0.5, FIG-VEC); label ≈6.6 (inconsistent with curve) (FIG-LBL) | FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |
| TaNbV67HfZrTi33 | 4.3 | rho/chi (Table I) | 4.11 (Cp) | TAB | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| NbV67HfZrTi33 | 7.2 | rho/chi |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| TaV67HfZrTi33 | 4 | rho/chi |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| TaMoScV67HfZrTi33 | — | no transition above 1.9 K |  | SEC | upper bound (Tc < 1.9 K) | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**41. von Rohr i in., *Physical Review Materials* 2, 034801 (2018)** — [10.1103/PhysRevMaterials.2.034801](https://doi.org/10.1103/PhysRevMaterials.2.034801)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNbHfZrTi_PRMat2018 | 7.6 | χ (line intersection) |  | TAB (Table I caption, '≈7.6 K') | baseline – corrected | Tc_baseline.xlsx row 10 |
| NbY0.33Mo0.67HfZrTi | 4.7 | χ ZFC (line intersection) |  | TAB (Table I; cross-checked with summary xls SUB-00x) | baseline – corrected | Tc_baseline.xlsx row 11 |
| TaNbHfY0.67Mo0.33Ti | 6.7 | χ ZFC (line intersection) |  | TAB (Table I; cross-checked with summary xls SUB-00x) | baseline – corrected | Tc_baseline.xlsx row 22 |
| TaNbHfZrY0.67Mo0.33 | 7.5 | χ ZFC (line intersection) |  | TAB (Table I; cross-checked with summary xls SUB-00x) | baseline – corrected | Tc_baseline.xlsx row 23 |
| TaNbY0.67Mo0.33ZrTi | 7.6 | χ ZFC (line intersection) |  | TAB (Table I; cross-checked with summary xls SUB-00x) | baseline – corrected | Tc_baseline.xlsx row 24 |
| TaY0.33Mo0.67HfZrTi | 3.5 | χ ZFC (line intersection) |  | TAB (Table I; cross-checked with summary xls SUB-00x) | baseline – corrected | Tc_baseline.xlsx row 25 |
| NbSc0.33Mo0.67HfZrTi | 4.4 | χ ZFC (line intersection) |  | TAB (Table I, verified arXiv 1708.02452) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| TaSc0.33Mo0.67HfZrTi | 2.9 | χ ZFC (line intersection) |  | TAB (Table I, verified arXiv 1708.02452) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| TaNbSc0.67Mo0.33ZrTi | 7.5 | χ ZFC (line intersection) |  | TAB (Table I, verified arXiv 1708.02452) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| TaNbHfSc0.67Mo0.33Ti | 6.6 | χ ZFC (line intersection) |  | TAB (Table I, verified arXiv 1708.02452) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |
| TaNbHfZrSc0.67Mo0.33 | 7.5 | χ ZFC (line intersection) |  | TAB (Table I, verified arXiv 1708.02452) | candidate | 0_compare_with_known.sh (never computed / not in Tc_baseline.xlsx) |

**42. Vrtnik i in., JALCOM 695,3530 (2017)** — [10.1016/j.jallcom.2016.11.417](https://doi.org/10.1016/j.jallcom.2016.11.417)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ta20Nb21Hf20Zr20Ti19 | 6 | Cp |  | SEC (two compilations agree) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**43. Wang i in., J. Low Temp. Phys. 32, 481 (1978)** — [10.1007/BF00117965](https://doi.org/10.1007/BF00117965)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| V9.5Nb29.8Ta60.7 | 4.47 | Cp |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| V21.2Nb42Ta36.8 | 4.41 | Cp |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |
| V25.1Nb57.8Ta17.1 | 4.67 | Cp |  | SEC | candidate | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**44. Wu i in., Nat. Sci. 10, 110 (2018)** — [10.4236/ns.2018.103012](https://doi.org/10.4236/ns.2018.103012)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNbHfZrTi_equimolar | 7.12 | ρ (50%), as-cast | 6.69 (ρ, homogenized) | TAB (verified 2026-10-19) | baseline | Tc_baseline.xlsx row 34 |
| NbTaTiZr_Wu2018 | 8.29 | ρ, as-cast | 8.27 (ρ, homogenized) | TAB (verified) | caution – inhomogeneous | lit search 2026-10-19 (literature_baseline_extension.md) |
| NbTaTiZrHfV_Wu2018 | 5.1 | unknown |  | SEC | caution – secondary source | V additions 2026-10-23 (Tc_literature_V_additions.csv) |

**45. Wu i in., PRL 132,166002 (2024)** — [10.1103/PhysRevLett.132.166002](https://doi.org/10.1103/PhysRevLett.132.166002)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TaNbHfZr_Wu2024 | 8.1 | ambient-pressure value |  | SEC | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**46. Yuan i in., Front.Mater. 5,72 (2018)** — [10.3389/fmats.2018.00072](https://doi.org/10.3389/fmats.2018.00072)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti15Zr15Nb35Ta35 | 8 | ρ and Cp | 8.5 (χ onset) | TAB (verified, full text) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

**47. Zeng i in., *Advanced Quantum Technologies* 6, 2300213 (2023)** — [10.1002/qute.202300213](https://doi.org/10.1002/qute.202300213)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| TiHfNbTaMo | 3.42 |  |  | TAB per summary xls (BULK-006) | baseline | Tc_baseline.xlsx row 29 |

**48. Zeng i in., Sci.China PMA 66,277412 (2023)** — [10.1007/s11433-023-2113-6](https://doi.org/10.1007/s11433-023-2113-6)

| stop | Tc [K] | metoda | Tc alt | źródło wartości | status | pochodzenie |
|---|---|---|---|---|---|---|
| Ti25Hf25Nb25Ta25 | 6.3 | Cp | 6.75 (χ); 6.77 (ρ) | TAB (verified, arXiv 2308.14612) | candidate | lit search 2026-10-19 (literature_baseline_extension.md) |

### 3.2 `results/Tc_literature_V_additions.csv` — wpisy spoza arkusza (stopy dwuskładnikowe z V, czysty V)

Używane przez `bash.old/0_compare_with_known_V.sh` (`BINARIES=1`); KKR GL jeszcze niepoliczone.

**—, Kittel / Roberts 1976**

| stop | Tc [K] | metoda | θ_D [K] | γ [mJ/mol K²] | źródło wartości | uwagi |
|---|---|---|---|---|---|---|
| V | 5.4 | Cp/chi (element) | 380 | 9.26 | SEC | pure bcc V reference point |

**Corsan, phys. stat. sol. 40, 657 (1970)** — [10.1002/pssb.19700400225](https://doi.org/10.1002/pssb.19700400225)

| stop | Tc [K] | metoda | θ_D [K] | γ [mJ/mol K²] | źródło wartości | uwagi |
|---|---|---|---|---|---|---|
| V85Nb15 | 4.3 | Cp | 364 | 9.2 | SEC | via Senga 2026 Table 2 |
| V70Nb30 | 4.03 | Cp | 326 | 8.7 | SEC |  |
| V50Nb50 | 4.25 | Cp | 298 | 8.3 | SEC |  |
| V31Nb69 | 4.9 | Cp | 285 | 8 | SEC |  |
| V20Nb80 | 5.76 | Cp | 275 | 7.9 | SEC |  |
| V10Nb90 | 6.95 | Cp | 266 | 7.8 | SEC |  |
| V90Ta10 | 4.47 | Cp | 341 | 8.8 | SEC |  |
| V76Ta24 | 3.58 | Cp | 300 | 8.5 | SEC |  |
| V50Ta50 | 2.73 | Cp | 273 | 7.6 | SEC |  |
| V30Ta70 | 3 | Cp | 256 | 6.4 | SEC |  |
| V10Ta90 | 3.58 | Cp | 250 | 5.7 | SEC |  |

**Cheng, Phys. Rev. 126, 2030 (1962)** — [10.1103/PhysRev.126.2030](https://doi.org/10.1103/PhysRev.126.2030)

| stop | Tc [K] | metoda | θ_D [K] | γ [mJ/mol K²] | źródło wartości | uwagi |
|---|---|---|---|---|---|---|
| Ti70V30 | 6.14 | Cp | 244 | 10 | SEC | via Senga 2026 Table 2 |
| Ti50V50 | 7.3 | Cp | 262 | 10.8 | SEC |  |
| Ti25V75 | 7.16 | Cp | 279 | 10.6 | SEC |  |
| Ti15V85 | 7.02 | Cp | 283 | 10.3 | SEC |  |

**Kuentzler, Solid State Commun. 59, 299 (1986)** — [10.1016/0038-1098(86)90412-6](https://doi.org/10.1016/0038-1098(86)90412-6)

| stop | Tc [K] | metoda | θ_D [K] | γ [mJ/mol K²] | źródło wartości | uwagi |
|---|---|---|---|---|---|---|
| V95Mo5 | 3.73 | Cp | 430 | 8.85 | SEC |  |
| V90Mo10 | 2.5 | Cp | 417 | 7.55 | SEC |  |

### 3.3 `papers_data/bcc_hea_superconducting_tc_summary.xls` — rekordy spoza arkusza

| stop | Tc [K] | definicja Tc | praca | DOI | status w zestawieniu | uwagi |
|---|---|---|---|---|---|---|
| Ta34Nb33Hf8Zr14Ti11  | 7.3 | Bulk superconducting transition | Koželj, *Physical Review Letters* 113, 107001 (2014) | 10.1103/PhysRevLett.113.107001 | Included | Local lattice distortion may be present although diffraction gives an average BCC structure. |
| (TaNb)1-x(ZrHfTi)x x=0.88 | 2.77 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.76 | 4.61 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.65 | 5.6 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.54 | 6.14 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.43 | 6.76 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.33 | 6.43 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included as thin-film data | Thin-film pseudo-BCC sample. |
| (TaNb)1-x(ZrHfTi)x x=0.21 | 6.23 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included with thin-film and HEA-definition caveats | Thin-film pseudo-BCC sample. Composition falls outside a strict 5-35 at.% per-element HEA range. |
| (TaNb)1-x(ZrHfTi)x x=0.13 | 6.02 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included with thin-film and HEA-definition caveats | Thin-film pseudo-BCC sample. Composition falls outside a strict 5-35 at.% per-element HEA range. |
| (TaNb)1-x(ZrHfTi)x x=0.04 | 5.57 | Zero-field transport Tc(0) | Zhang, *Physical Review Research* 2, 013375 (2020) | 10.1103/PhysRevResearch.2.013375 | Included with thin-film and HEA-definition caveats | Thin-film pseudo-BCC sample. Composition falls outside a strict 5-35 at.% per-element HEA range. |
| Hf5Nb45Sc10Ti5Zr35  | 9.93 | Maximum reported Tc after 800 C anneal | , *arXiv preprint / Journal of Alloys and Compounds record*  (2026) |  | Excluded from strict single-phase-BCC subset | All elements are allowed, but the sample is eutectic/multiphase and Nb=45 at.%; retained only as a c |
| Mo35WxReyRuzPd5 series  | — | Superconductivity reported across the structural series | , *Scripta Materialia* Article 115986 (2024) | 10.1016/j.scriptamat.2024.115986 | Excluded | Excluded because Pd is not in the permitted element set. No strict Ru-containing BCC match using onl |


### 3.4 Czyste pierwiastki (`results/Tc_baseline.xlsx`, wiersze 41–52)

| pierwiastek | Tc [K] | uwaga w arkuszu |
|---|---|---|
| Nb | 9.25 | McMillan 1968 |
| Ta | 4.47 | McMillan 1968 |
| Mo | 0.92 | McMillan 1968 |
| W | 0.011 | McMillan 1968 |
| La | 4.87 | faza dhcp |
| Re | 1.7 | hcp |
| Ru | 0.49 | hcp |
| Zr | 0.55 | hcp α-Zr |
| Ti | 0.4 | hcp α-Ti |
| Hf | 0.128 | hcp |
| Sc | — | brak nadprzewodnictwa przy ciśnieniu otoczenia |
| Y | — | brak nadprzewodnictwa przy ciśnieniu otoczenia |
| V | 5.4 | Kittel / Roberts 1976 (`Tc_literature_V_additions.csv`) |

- McMillan 1968 = W. L. McMillan, *Phys. Rev.* **167**, 331 (1968), doi [10.1103/PhysRev.167.331](https://doi.org/10.1103/PhysRev.167.331).
- Pozostałe wartości to standardowe dane tablicowe; źródło w arkuszu nie jest podane.
- Tylko Nb, Ta, Mo, W i V są w fazie bcc; dla pozostałych pierwiastków wartości dotyczą faz hcp/dhcp i nie są
  bezpośrednim punktem odniesienia dla bcc KKR.

### 3.5 Stopy dwuskładnikowe — odniesienia używane w notatkach (wartości NIE są w arkuszu)

Wartości z `VALIDATION_NOTE.md` i z notatek sesji 2026-10-22/23. Przytaczane z pamięci; **przed publikacją trzeba je
zweryfikować w źródłach**.

| linia | Tc exp (maksimum) | źródło podawane w notatkach |
|---|---|---|
| Nb–Zr | ok. 11 K przy Nb75Zr25 | J. K. Hulm, R. D. Blaugher, *Phys. Rev.* **123**, 1569 (1961); Corsan 1968 |
| Nb–Ti | 9.88 K przy Nb70Ti30 (ok. 9.8–10 K dla Nb65–75) | Suenaga & Ralls; praca doktorska Mullera; Hulm & Blaugher 1961 |
| Nb–Hf | ok. 9.5–10 K przy Hf20–25 | Hulm & Blaugher 1961 |
| Nb–Ta | monotonicznie 9.2 → 4.5 K | Hulm & Blaugher 1961 |
| Mo–Re | ok. 12–13 K przy ok. Mo60Re40 | Hulm & Blaugher 1961 / Matthias (drugie maksimum e/a ≈ 6.4) |
| Mo–Tc | ok. 15 K | Matthias |
| Ti–V | 6.1–7.3 K | Cheng i in., *Phys. Rev.* **126**, 2030 (1962) — w 3.2, z θ_D i γ |
| Nb50Ti50 | ok. 9–10 K | walidacja NbTiZrHf (`3_run_validation_NbTiZrHf.sh`); źródło w skrypcie nie jest podane |

- Skrypt `3_run_validation_binaries*.sh` cytuje Hulm & Blaugher (1961) i Matthiasa jako źródło przebiegów Tc(x).
- Linie: Nb–Zr, Nb–Ti, Mo–Re, Nb–Ta, Nb–Mo, Nb–Hf.
