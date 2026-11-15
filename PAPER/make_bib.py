"""Build PAPER/refs.bib from DOIs only (no hand-typed bibliographic data).

Every entry is fetched from doi.org (content negotiation, application/x-bibtex) and stored under the key given in
DOIS, so each citation in main.tex / supplement.tex resolves to a real, existing record.  Entries that cannot be
fetched are listed at the end and must be verified by hand (they are written to refs_missing.txt).

    python PAPER/make_bib.py            # writes PAPER/refs.bib (+ refs_missing.txt)
"""
import re
import subprocess
import sys
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# key -> DOI (methods, physics, HEA reviews)
DOIS = {
    # superconductivity theory used in the pipeline
    'McMillan1968': '10.1103/PhysRev.167.331',
    'AllenDynes1975': '10.1103/PhysRevB.12.905',
    'GaspariGyorffy1972': '10.1103/PhysRevLett.28.801',
    'Papaconstantopoulos1977': '10.1103/PhysRevB.15.4221',
    'Butler1977': '10.1103/PhysRevB.15.5267',
    'Hopfield1969': '10.1103/PhysRev.186.443',
    'Eliashberg1960': None,                       # Sov. Phys. JETP 11, 696 (1960) - no DOI
    'Dynes1972': '10.1016/0038-1098(72)90603-5',
    'Morel1962': '10.1103/PhysRev.125.1263',
    'Matthias1955': '10.1103/PhysRev.97.74',
    'Hulm1961': '10.1103/PhysRev.123.1569',
    # KKR-CPA
    'Korringa1947': '10.1016/0031-8914(47)90013-X',
    'KohnRostoker1954': '10.1103/PhysRev.94.1111',
    'Soven1967': '10.1103/PhysRev.156.809',
    'Akai1989': '10.1088/0953-8984/1/43/006',
    'Lloyd1967': '10.1088/0370-1328/90/1/325',
    'PerdewBurkeErnzerhof1996': '10.1103/PhysRevLett.77.3865',
    'Jaskiewicz2016': '10.1002/pssr.201600056',
    'Jaskiewicz2019': '10.1103/PhysRevB.100.184503',
    'Rajput1996': '10.1103/PhysRevB.53.4950',
    # DFT for superconductors (expensive route) and ML for Tc
    'Giustino2017': '10.1103/RevModPhys.89.015003',
    'Baroni2001': '10.1103/RevModPhys.73.515',
    'Stanev2018': '10.1038/s41524-018-0085-8',
    'Xie2022': '10.1038/s41524-021-00666-7',
    # spin fluctuations (V, Ti-V)
    'Matin2014': '10.1140/epjb/e2014-50036-2',
    'RietschelWinter1979': '10.1103/PhysRevLett.43.1256',
    'Bose2009': '10.1088/0953-8984/21/2/025602',
    'Berk1966': '10.1103/PhysRevLett.17.433',
    # HEA
    'Yeh2004': '10.1002/adem.200300567',
    'Cantor2004': '10.1016/j.msea.2003.10.257',
    'Kozelj2014': '10.1103/PhysRevLett.113.107001',
    'SunCava2019': '10.1103/PhysRevMaterials.3.090301',
    'Kitagawa2020': '10.3390/met10081078',
    # ML / optimisation tools
    'Prokhorenkova2018': '10.48550/arXiv.1706.09516',
    'Lakshminarayanan2017': '10.48550/arXiv.1612.01474',
    'Srinivas2010': '10.48550/arXiv.0912.3995',
    'Sobol1967': '10.1016/0041-5553(67)90144-9',
    'Ward2016': '10.1038/npjcompumats.2016.28',      # composition-derived descriptors (Magpie)
    'Zhang2008': '10.1002/adem.200700240',           # HEA parameters: size mismatch, mixing entropy
    # binaries
    'Kushwaha2025': '10.48550/arXiv.2512.18821',
    'Sitaraman2022': '10.48550/arXiv.2208.10678',
    'Junod1986': '10.1007/BF00683466',
    # Coulomb pseudopotential values
    'Savrasov1996': '10.1103/PhysRevB.54.16487',
    'Carbotte1990': '10.1103/RevModPhys.62.1027',
    # elastic constants -> Debye temperature
    'Hill1952': '10.1088/0370-1298/65/5/307',
    'Anderson1963': '10.1016/0022-3697(63)90067-2',
}


# Sources without a DOI: typed by hand after reading the document itself (keep this list short).
MANUAL = [
    # PhD thesis, Table 2 (p. 39) read from the scanned page on 2026-11-12
    "@phdthesis{Muller1989, author={Muller, H. J.}, title={The Upper Critical Field of Niobium-Titanium}, school={University of Wisconsin--Madison}, year={1989}, url={https://fs.magnet.fsu.edu/~lee/asc/pdf_papers/theses/hjm89phd.pdf}}",
]


def fetch(doi):
    r = subprocess.run(['curl', '-s', '-L', '--max-time', '30', '-H', 'Accept: application/x-bibtex',
                        f'https://doi.org/{doi}'], capture_output=True, text=True)
    txt = r.stdout.strip()
    return txt if txt.startswith('@') else None


def rekey(entry, key):
    return re.sub(r'^@(\w+)\{[^,]*,', lambda m: f'@{m.group(1)}{{{key},', entry, count=1)


def sanitise(entry):
    """Make the doi.org BibTeX safe for pdflatex: drop month fields, escape &, replace non-Latin characters."""
    entry = re.sub(r',\s*month=\w+', '', entry)
    entry = entry.replace('≤', '$\\leq$').replace('≥', '$\\geq$').replace('−', '-').replace('–', '--').replace('—', '--')
    entry = re.sub(r'(?<!\\)&', r'\\&', entry)
    entry = re.sub(r'[\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef]+', '', entry)   # CJK / Hangul glyphs added to some author names
    # MathML in titles: keep the text of the leaves, drop the markup
    entry = re.sub(r'<mml:math[^>]*>(.*?)</mml:math>',
                   lambda m: ' ' + re.sub(r'<[^>]+>', '', m.group(1)).replace('\n', '').strip() + ' ', entry, flags=re.S)
    entry = re.sub(r'<sub>(.*?)</sub>', r'$_{\1}$', entry)
    entry = re.sub(r'<sup>(.*?)</sup>', r'$^{\1}$', entry)
    entry = re.sub(r'<[^>]+>', '', entry)
    for a, b in [('“', '"'), ('”', '"'), ('’', "'"), ('‘', "'"), ('‐', '-'), ('\u2009', ' '), ('\u00a0', ' '), ('…', '...')]:
        entry = entry.replace(a, b)
    entry = re.sub(r'\s*\n\s*', ' ', entry)
    entry = re.sub(r'  +', ' ', entry).replace('{ ', '{').replace(' }', '}')
    if '10.48550/' in entry.upper() or '10.48550/ARXIV' in entry:
        entry = re.sub(r'^@\w+\{', '@misc{', entry)
    return entry


def literature_dois():
    """DOIs of the experimental papers in results/Tc_literature.xlsx, keyed FirstAuthor + short DOI hash."""
    ws = openpyxl.load_workbook(ROOT / 'results/Tc_literature.xlsx', data_only=True)['Tc_literature']
    out = {}
    for r in range(4, ws.max_row + 1):
        doi, author = ws.cell(r, 15).value, ws.cell(r, 26).value
        if not doi or not author:
            continue
        doi = str(doi).strip()
        if doi.lower().startswith('arxiv:'):
            doi = '10.48550/arXiv.' + doi.split(':', 1)[1]
        key = 'lit_' + re.sub(r'[^A-Za-z]', '', str(author)) + '_' + re.sub(r'[^A-Za-z0-9]', '', doi.split('/')[-1])[-8:]
        out[key] = doi
    return out


def main():
    all_dois = dict(DOIS)
    all_dois.update(literature_dois())
    entries, missing = [], []
    for key, doi in all_dois.items():
        if doi is None:
            missing.append((key, 'no DOI'))
            continue
        e = fetch(doi)
        if e is None:
            missing.append((key, doi))
            print('MISSING', key, doi, file=sys.stderr)
            continue
        entries.append(sanitise(rekey(e, key)))
        print('ok', key)
    (HERE / 'refs.bib').write_text('% generated by make_bib.py from DOIs (doi.org content negotiation); do not edit by hand\n'
                                   + '\n\n'.join(entries + MANUAL) + '\n')
    (HERE / 'refs_missing.txt').write_text('\n'.join(f'{k}\t{d}' for k, d in missing) + '\n')
    print(f'{len(entries)} entries written, {len(missing)} missing -> refs_missing.txt')


if __name__ == '__main__':
    main()
