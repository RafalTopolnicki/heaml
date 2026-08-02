#!/bin/bash
ew=0.6
params="--overwrite_params --ew ${ew} --rel sra --bzqlty 10"
outputdir="results/comparison.sra.ew${ew}"

# Jaskiewicz 2019
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Ti --concentrations 0.335 0.335 0.11 0.11 0.11 --workdir ${outputdir}/TaNb67HfZrTi33 ${params}

# Pure elements
#python scripts/process_hea.py --element_labels Ta --concentrations 100 --workdir ${outputdir}/Ta/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Nb --concentrations 100 --workdir ${outputdir}/Nb/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Hf --concentrations 100 --workdir ${outputdir}/Hf/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Zr --concentrations 100 --workdir ${outputdir}/Zr/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Ti --concentrations 100 --workdir ${outputdir}/Ti/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Sc --concentrations 100 --workdir ${outputdir}/Sc/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels W --concentrations 100 --workdir ${outputdir}/W/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Y --concentrations 100 --workdir ${outputdir}/Y/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels La --concentrations 100 --workdir ${outputdir}/La/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Mo --concentrations 100 --workdir ${outputdir}/Mo/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Re --concentrations 100 --workdir ${outputdir}/Re/ --overwrite_params ${params}
#python scripts/process_hea.py --element_labels Ru --concentrations 100 --workdir ${outputdir}/Ru/ --overwrite_params ${params}





## Jaskiewicz 2016
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Ti --concentrations 34 33 8 14 11 --workdir ${outputdir}/Ta34Nb33Hf8Zr14Ti11 ${params}


## https://journals.aps.org/prb/pdf/10.1103/fnz7-bzmw
#python scripts/process_hea.py --element_labels Nb Hf Ti Zr --concentrations 37.5 37.5 12.5 12.5 --workdir ${outputdir}/NbHf75TiZr25 ${params}
#python scripts/process_hea.py --element_labels Nb Zr Ti Hf --concentrations 37.5 37.5 12.5 12.5 --workdir ${outputdir}/NbZr75TiHf25 ${params}
#python scripts/process_hea.py --element_labels Nb Ti Zr Hf --concentrations 37.5 37.5 12.5 12.5 --workdir ${outputdir}/NbTi75ZrHf25 ${params}

## https://www.mdpi.com/1996-1944/16/17/5814
#python scripts/process_hea.py --element_labels Nb Ti Zr Ta Hf --concentrations 34 33 14 11 8 --workdir ${outputdir}/Nb34Ti33Zr14Ta11Hf8 ${params}

##https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.184512
#python scripts/process_hea.py --element_labels Nb Ta Mo Hf W --concentrations 33.5 33.5 11 11 11 --workdir ${outputdir}/NbTa67MoHfW33 ${params}


## https://doi.org/10.3390/ma15031122 — Sc-containing quaternary/quinary/senary BCC HEAs
#python scripts/process_hea.py --element_labels Sc Hf Nb Ti    --concentrations 20 25 30 25          --workdir ${outputdir}/ScHfNbTi         ${params}
#python scripts/process_hea.py --element_labels Sc Hf Nb Zr    --concentrations 21 26 28 25          --workdir ${outputdir}/ScHfNbZr         ${params}
#python scripts/process_hea.py --element_labels Sc Hf Ta Ti    --concentrations 12 24 38 26          --workdir ${outputdir}/ScHfTaTi         ${params}
#python scripts/process_hea.py --element_labels Sc Nb Ti Zr    --concentrations 18 32 24 26          --workdir ${outputdir}/ScNbTiZr         ${params}
#python scripts/process_hea.py --element_labels Sc Ta Ti Zr    --concentrations 4  64 22 10          --workdir ${outputdir}/ScTaTiZr         ${params}
#python scripts/process_hea.py --element_labels Sc Hf Nb Ta Ti --concentrations 8  22 25 25 20       --workdir ${outputdir}/ScHfNbTaTi       ${params}
#python scripts/process_hea.py --element_labels Sc Hf Nb Ta Zr --concentrations 9  21 26 26 18       --workdir ${outputdir}/ScHfNbTaZr_bccL  ${params}
#python scripts/process_hea.py --element_labels Sc Hf Nb Ta Zr --concentrations 15 22 22 20 21       --workdir ${outputdir}/ScHfNbTaZr_bccS  ${params}
#python scripts/process_hea.py --element_labels Sc Nb Ta Ti Zr --concentrations 4  29 41 15 11       --workdir ${outputdir}/ScNbTaTiZr       ${params}
#python scripts/process_hea.py --element_labels Sc Hf Nb Ta Ti Zr --concentrations 12 17 19 19 17 16 --workdir ${outputdir}/ScHfNbTaTiZr     ${params}


## https://doi.org/10.1103/PhysRevMaterials.2.034801 — Y/Mo substitution in TaNbHfZrTi
#python scripts/process_hea.py --element_labels Ti Nb Zr Hf Mo Y    --concentrations 11 33.5 11 11 22.445 11.055 --workdir ${outputdir}/NbY0.33Mo0.67HfZrTi   ${params}
#python scripts/process_hea.py --element_labels Ti Zr Hf Ta Mo Y    --concentrations 11 11 11 33.5 22.445 11.055 --workdir ${outputdir}/TaY0.33Mo0.67HfZrTi   ${params}
#python scripts/process_hea.py --element_labels Ti Nb Zr Ta Mo Y    --concentrations 11 33.5 11 33.5 3.63 7.37   --workdir ${outputdir}/TaNbY0.67Mo0.33ZrTi   ${params}
#python scripts/process_hea.py --element_labels Ti Nb Hf Ta Mo Y    --concentrations 11 33.5 11 33.5 3.63 7.37   --workdir ${outputdir}/TaNbHfY0.67Mo0.33Ti   ${params}
#python scripts/process_hea.py --element_labels Nb Zr Hf Ta Mo Y    --concentrations 33.5 11 11 33.5 3.63 7.37   --workdir ${outputdir}/TaNbHfZrY0.67Mo0.33  ${params}
#python scripts/process_hea.py --element_labels Ti Nb Zr Hf Ta      --concentrations 11 33.5 11 11 33.5          --workdir ${outputdir}/TaNbHfZrTi_PRMat2018  ${params}


## ── NEW ENTRIES FROM bcc_hea_superconducting_tc_summary.xls ─────────────────


## Marik 2018, J. Alloys Compd. 769, 1059 | doi:10.1016/j.jallcom.2018.08.039
## Nb20Re20Zr20Hf20Ti20 | Tc=5.3 K (bulk, transport+magnetization+heat capacity) | equiatomic 5-element BCC
#python scripts/process_hea.py --element_labels Nb Re Zr Hf Ti --concentrations 20 20 20 20 20 --workdir ${outputdir}/NbReZrHfTi ${params}
#
#
### Kim 2020, Acta Mater. 186, 250 | doi:10.1016/j.actamat.2020.01.007
## Ta(1/6)Nb(2/6)Hf(1/6)Zr(1/6)Ti(1/6) | Tc=7.9 K (resistive onset), 7.8 K (zero resistance) | single BCC phase
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Ti --concentrations 16.6667 33.3333 16.6667 16.6667 16.6667 --workdir ${outputdir}/TaNb2HfZrTi ${params}
#
#
### Kitagawa 2022, J. Alloys Compd. 924, 166473 | doi:10.1016/j.jallcom.2022.166473
## HfMoNbTiZr equiatomic | Tc=4.1 K (bulk) | single BCC phase
#python scripts/process_hea.py --element_labels Hf Mo Nb Ti Zr --concentrations 20 20 20 20 20 --workdir ${outputdir}/HfMoNbTiZr ${params}
#
#
### Zeng 2023, Adv. Quantum Technol. 6, 2300213 | doi:10.1002/qute.202300213
## TiHfNbTaMo equiatomic | Tc=3.42 K (experimental) | single homogeneous BCC phase (Im-3m), a=3.445 A
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Mo --concentrations 20 20 20 20 20 --workdir ${outputdir}/TiHfNbTaMo ${params}
#
#
### Motla 2022, Phys. Rev. B 105, 144501 | doi:10.1103/PhysRevB.105.144501
## Nb60Re10Zr10Hf10Ti10 | Tc=5.7 K (bulk, muSR+resistivity+magnetization) | single centrosymmetric BCC (Im-3m)
## CAVEAT: Nb=60 at.% is outside strict HEA definition (5-35 at.% per element); included for methodology testing
#python scripts/process_hea.py --element_labels Nb Re Zr Hf Ti --concentrations 60 10 10 10 10 --workdir ${outputdir}/Nb60Re10Zr10Hf10Ti10 ${params}
#
#
### Chakrabarty 2025, J. Appl. Phys. 137, 215901 | doi:10.1063/5.0265943
## (NbTa)0.55(HfTiZr)0.45 | Tc=7.2 K ambient pressure (max 10.1 K at 23.6 GPa) | single-phase BCC
#python scripts/process_hea.py --element_labels Nb Ta Hf Ti Zr --concentrations 27.5 27.5 15 15 15 --workdir ${outputdir}/NbTa55HfTiZr45 ${params}


## Hattori 2023, J. Alloys Metall. Syst. 3, 100020 | arXiv:2307.01958
## Ti-Hf-Nb-Ta-Re composition series | Tc from specific heat
## RE-001: Ti35Hf25Nb25Ta5Re10 | Tc=3.95 K | single BCC phase
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Re --concentrations 35 25 25 5 10 --workdir ${outputdir}/Ti35Hf25Nb25Ta5Re10 ${params}
## RE-002: Ti30Hf20Nb35Ta5Re10 | Tc=4.38 K | CAVEAT: weak segregation into two BCC phases
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Re --concentrations 30 20 35 5 10 --workdir ${outputdir}/Ti30Hf20Nb35Ta5Re10 ${params}
## RE-003: Ti25Hf15Nb35Ta15Re10 | Tc=4.1 K | CAVEAT: weak segregation into two BCC phases
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Re --concentrations 25 15 35 15 10 --workdir ${outputdir}/Ti25Hf15Nb35Ta15Re10 ${params}
## RE-004: Ti20Hf10Nb35Ta25Re10 | Tc=3.62 K | CAVEAT: two compositionally distinct BCC phases
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Re --concentrations 20 10 35 25 10 --workdir ${outputdir}/Ti20Hf10Nb35Ta25Re10 ${params}
## RE-005: Ti15Hf5Nb35Ta35Re10 | Tc=3.25 K | CAVEAT: two compositionally distinct BCC phases
#python scripts/process_hea.py --element_labels Ti Hf Nb Ta Re --concentrations 15 5 35 35 10 --workdir ${outputdir}/Ti15Hf5Nb35Ta35Re10 ${params}


## von Rohr 2018, Phys. Rev. Mater. 2, 034801 | doi:10.1103/PhysRevMaterials.2.034801
## Sc/Mo isoelectronic substitutions in TaNbHfZrTi (same paper as Y/Mo above)
# SUB-006: [{Sc0.33Mo0.67}Nb]0.67(HfZrTi)0.33 — Sc0.33Mo0.67 replaces Ta | Tc=4.4 K
#python scripts/process_hea.py --element_labels Sc Mo Nb Hf Zr Ti --concentrations 11.055 22.445 33.5 11 11 11 --workdir ${outputdir}/NbSc0.33Mo0.67HfZrTi ${params}
## SUB-007: [Ta{Sc0.33Mo0.67}]0.67(HfZrTi)0.33 — Sc0.33Mo0.67 replaces Nb | Tc=2.9 K
#python scripts/process_hea.py --element_labels Ta Sc Mo Hf Zr Ti --concentrations 33.5 11.055 22.445 11 11 11 --workdir ${outputdir}/TaSc0.33Mo0.67HfZrTi ${params}
## SUB-008: [TaNb]0.67({Sc0.67Mo0.33}ZrTi)0.33 — Sc0.67Mo0.33 replaces Hf | Tc=7.5 K
#python scripts/process_hea.py --element_labels Ta Nb Sc Mo Zr Ti --concentrations 33.5 33.5 7.37 3.63 11 11 --workdir ${outputdir}/TaNbSc0.67Mo0.33ZrTi ${params}
## SUB-009: [TaNb]0.67(Hf{Sc0.67Mo0.33}Ti)0.33 — Sc0.67Mo0.33 replaces Zr | Tc=6.6 K
#python scripts/process_hea.py --element_labels Ta Nb Hf Sc Mo Ti --concentrations 33.5 33.5 11 7.37 3.63 11 --workdir ${outputdir}/TaNbHfSc0.67Mo0.33Ti ${params}
## SUB-010: [TaNb]0.67(HfZr{Sc0.67Mo0.33})0.33 — Sc0.67Mo0.33 replaces Ti | Tc=7.5 K
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Sc Mo --concentrations 33.5 33.5 11 11 7.37 3.63 --workdir ${outputdir}/TaNbHfZrSc0.67Mo0.33 ${params}


## ── FROM: "High-Entropy Superconducting Materials" (IntechOpen review) ────────


## von Rohr et al. 2016, PNAS 113, E7144 — [TaNb]x[ZrHfTi]1-x composition series
## x=0.67 equiatomic already in script above (Jaskiewicz / TaNbHfZrTi_PRMat2018)
## x=0.70 | Tc=8.03 K | single BCC phase
#python scripts/process_hea.py --element_labels Ta Nb Zr Hf Ti --concentrations 35 35 10 10 10 --workdir ${outputdir}/TaNb70ZrHfTi30 ${params}
## x=0.60 | Tc=7.56 K | single BCC phase
#python scripts/process_hea.py --element_labels Ta Nb Zr Hf Ti --concentrations 30 30 13.3333 13.3333 13.3333 --workdir ${outputdir}/TaNb60ZrHfTi40 ${params}
## x=0.50 | Tc=6.46 K | single BCC phase
#python scripts/process_hea.py --element_labels Ta Nb Zr Hf Ti --concentrations 25 25 16.6667 16.6667 16.6667 --workdir ${outputdir}/TaNb50ZrHfTi50 ${params}
## x=0.16 | Tc=4.52 K | single BCC phase (Ti/Zr/Hf-rich extreme)
#python scripts/process_hea.py --element_labels Ta Nb Zr Hf Ti --concentrations 8 8 28 28 28 --workdir ${outputdir}/TaNb16ZrHfTi84 ${params}
#
#
### Wu et al. 2018, Natural Science 10, 110
## Ta20Nb20Hf20Zr20Ti20 equimolar | Tc=7.12 K | single BCC phase
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Ti --concentrations 20 20 20 20 20 --workdir ${outputdir}/TaNbHfZrTi_equimolar ${params}


### Sobota et al. 2022, Phys. Rev. B 106, 184512 | doi:10.1103/PhysRevB.106.184512
## (NbTa)0.67(MoHfW)0.33 | Tc=4.3 K | single BCC phase
#python scripts/process_hea.py --element_labels Nb Ta Mo Hf W --concentrations 33.5 33.5 11 11 11 --workdir ${outputdir}/NbTa67MoHfW33 ${params}


## Krnel et al. 2022, Materials 15, 1122 | doi:10.3390/ma15031122
## Equimolar Sc-substituted pentenary series (Sc replaces one element of TaNbHfZrTi)
## Ta20Nb20Hf20Sc20Ti20 (Zr→Sc) | Tc=6.60 K | Bc2(0)=13.1 T | BCC
#python scripts/process_hea.py --element_labels Ta Nb Hf Sc Ti --concentrations 20 20 20 20 20 --workdir ${outputdir}/TaNbHfScTi ${params}
## Ta20Nb20Hf20Zr20Sc20 (Ti→Sc) | Tc=7.70 K | Bc2(0)=12.4 T | BCC
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Sc --concentrations 20 20 20 20 20 --workdir ${outputdir}/TaNbHfZrSc ${params}
## Ta20Nb20Sc20Zr20Ti20 (Hf→Sc) | Tc=7.90 K | Bc2(0)=19.3 T | BCC
#python scripts/process_hea.py --element_labels Ta Nb Sc Zr Ti --concentrations 20 20 20 20 20 --workdir ${outputdir}/TaNbScZrTi ${params}
## Ta16.67Nb16.67Hf16.67Zr16.67Ti16.67Sc16.67 equimolar senary | Tc=7.20 K | Bc2(0)=14.1 T | BCC
#python scripts/process_hea.py --element_labels Ta Nb Hf Zr Ti Sc --concentrations 16.6667 16.6667 16.6667 16.6667 16.6667 16.6667 --workdir ${outputdir}/TaNbHfZrTiSc ${params}
