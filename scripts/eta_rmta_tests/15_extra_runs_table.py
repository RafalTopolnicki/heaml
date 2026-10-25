"""Per-element N(E_F) and eta (A valence / B surface / G gg) for the extra final-SCF test runs (reltyp sra/nrl/srals, LDA/PBE, bz20/30):
alloy Ta34Nb33Hf8Zr14Ti11 at a=3.36 A, pure V, pure Nb at a=6.20. Prints a table vs Jaskiewicz 2016 / Papaconstantopoulos 1977 / Bose 2009."""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import fort51_path
from macmillan import parse_fort51
from macmillan_cutoff import compute_one_combination_cutoff
from src.consts import VALENCE_NODES_EXPECTED
T = 'results/test.nrl.ew0.6'; B = 'results/comparison.sra.ew0.6.bz10-10-20'
SETS = [('alloy Ta34Nb33Hf8Zr14Ti11', ['Ta', 'Nb', 'Hf', 'Zr', 'Ti'], [.34, .33, .08, .14, .11],
         [('SRA PBE bz20 (prod., a0 3.365)', f'{B}/Ta34Nb33Hf8Zr14Ti11'), ('SRA LDA', f'{T}/TNHZT_a3.36_sra_vbh'), ('NRL PBE', f'{T}/TNHZT_a3.36_nrl'),
          ('NRL LDA', f'{T}/TNHZT_a3.36_nrl_vbh'), ('SRA+LS (srals) PBE', f'{T}/TNHZT_a3.36_srals')],
         {'Ta': (17.2, 0.9, 50.7, 102.4), 'Nb': (18.8, 4.7, 55.3, 102.0), 'Hf': (16.2, 1.7, 71.4, 87.5), 'Zr': (17.2, 7.1, 79.3, 89.7), 'Ti': (26.6, 4.9, 45.7, 75.6)}, 'Jaskiewicz 2016'),
        ('pure V (a0 5.663)', ['V'], [1.0],
         [('SRA PBE bz20 (prod.)', f'{B}/V'), ('SRA PBE bz30', f'{T}/V_sra_pbe_bz30'), ('NRL LDA bz30', f'{T}/V_nrl_vbh_bz30'), ('srals PBE bz20', f'{T}/V_srals'), ('NRL PBE bz20 (V set)', f'{T}/V_set_nrl/V')],
         {'V': (25.4, 4.1, 33.8, 104.0)}, 'Papaconstantopoulos 1977 (Bose 2009 LR: N 25.3, eta 147.6)'),
        ('pure Nb (a0 6.20)', ['Nb'], [1.0],
         [('SRA PBE bz20 (prod., a0 6.27)', f'{B}/Nb'), ('SRA PBE bz30', f'{T}/Nb_a6.20_sra_pbe_bz30'), ('NRL LDA bz30', f'{T}/Nb_a6.20_nrl_vbh_bz30'), ('srals LDA bz20', f'{T}/Nb_a6.20_srals_bz20'), ('SRA PBE bz20 a0 6.20', f'{T}/Nb_a6.20_sra_pbe_bz20')],
         {'Nb': (19.4, 9.7, 56.1, 91.2)}, 'Papaconstantopoulos 1977 (Bose 2009 LR: N 19.9, eta 178.6)')]
for title, labels, conc, runs, ref, refname in SETS:
    print(f'\n=== {title}   [N(E_F) both spins 1/Ry; eta mRy/a0^2 sp/pd/df/total]   reference: {refname}')
    for el in labels:
        r = ref[el]; print(f'  {el:3s} ref            N={r[0]:5.1f}   {r[1]:5.1f}/{r[2]:6.1f}/{r[3]:6.1f}/{sum(r[1:]):6.1f}')
    for tag, wd in runs:
        try: d = parse_fort51(fort51_path(wd))
        except (FileNotFoundError, OSError): print(f'  -- {tag}: not available yet'); continue
        ef = float(np.atleast_1d(d['ef'])[0]); mix = {m: 0.0 for m in ('valence', 'surface', 'gg')}
        print(f'  -- {tag}  (E_F {ef:.4f} Ry, nspin {d["nspin"] if "nspin" in d else "?"})')
        for b in d['components']:
            if b['spin'] != 1: continue
            b['ef'] = ef; el = labels[b['component'] - 1]; out = []
            for m in ('valence', 'surface', 'gg'):
                r = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode=m, valence_nodes=VALENCE_NODES_EXPECTED.get(el))
                e = [1e3 * r[f'eta_cutoff_{c}'] for c in ('sp', 'pd', 'df')]; out.append(f'{m[0].upper()} {e[0]:5.1f}/{e[1]:6.1f}/{e[2]:6.1f}/{sum(e):6.1f}')
                mix[m] += conc[b['component'] - 1] * sum(e)
            print(f'     {el:3s} N={2*r["Ntot"]:5.2f}   ' + '   '.join(out))
        if len(labels) > 1: print('     eta_mix: ' + '  '.join(f'{m}: {v:.1f}' for m, v in mix.items()) + f'   (ref {sum(conc[i]*sum(ref[l][1:]) for i,l in enumerate(labels)):.1f})')
