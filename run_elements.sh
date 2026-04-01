#!/bin/bash
## Ti, Nb, Zr, Hf, Ta, Sc, Y, La, Mo, W
# run full calculations for pure elements
python scripts/process_hea.py --concentrations 100 --element_labels Ti --workdir results/elements/Ti/
python scripts/process_hea.py --concentrations 100 --element_labels Nb --workdir results/elements/Nb/
python scripts/process_hea.py --concentrations 100 --element_labels Zr --workdir results/elements/Zr/
python scripts/process_hea.py --concentrations 100 --element_labels Hf --workdir results/elements/Hf/
python scripts/process_hea.py --concentrations 100 --element_labels Ta --workdir results/elements/Ta/
python scripts/process_hea.py --concentrations 100 --element_labels Sc --workdir results/elements/Sc/