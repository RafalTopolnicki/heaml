"""Build a slim init dataset (for run_opt_compositions.py --initdir) from a full KKR dataset, with eta/lambda/Tc
recomputed in the GL variant (Gaspari-Gyorffy surface formula, cutoff_mode='gg', with the Lloyd total DOS in 1/N,
dos_normalization='lloyd').  No gz archives are read later: each slim entry holds only the four small files that
run_opt_compositions.py (via process_kkr) needs:
    <entry>/results.json, run_params.json, debye/debye_summary.json, finalscf/mcmillan_cutoff_results.csv
plus the top-level parameters*.json.  fort.51 and finalscf_*.out are read ONCE here (plain or .gz) to rebuild
mcmillan_cutoff_results.csv; results.json is regenerated with process_kkr, so lambda / Tc_mu* / Stoner / features
are all consistent with the new eta.  Nothing in SRC is modified.

    python scripts/make_slim_dataset.py SRC DST [--elements Ti,Nb,...] [--workers 12]
    python scripts/make_slim_dataset.py SRC DST --finish        # after the SLURM finalscf rerun (see below)

--elements: keep only entries whose non-zero components all belong to the list (new element combinations are
            built this way from a larger dataset; run_params / results keep their original labels).
Entries without fort.51 or finalscf output in SRC cannot be converted; they are listed in DST/rerun_finalscf.csv
(name, Z list, concentrations, a0, KKR params) for slurm/topola_finalscf_rerun.sbatch, which writes a fresh
fort.51 + .out into DST/<entry>/finalscf; then `--finish` converts them and removes the bulky files again.
DST/slim_info.json records the settings and counts."""
import argparse, csv, glob, gzip, json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from macmillan_cutoff import run_mcmillan_cutoff_sweep        # noqa: E402
from src.process_kkr import process_kkr                       # noqa: E402
from src.elements import ELEMENTS                             # noqa: E402

KEEP = ['run_params.json', 'debye/debye_summary.json']
BULKY = ['fort.50', 'fort.51', 'mcmillan_cutoff_run.log']


def _first(pattern):
    m = sorted(glob.glob(pattern)); return m[0] if m else None


def _copy_plain(src, dst):
    """Copy src to dst, decompressing when src ends with .gz (dst gets the name without .gz)."""
    if src.endswith('.gz'):
        with gzip.open(src, 'rb') as f, open(dst[:-3] if dst.endswith('.gz') else dst, 'wb') as g: shutil.copyfileobj(f, g)
    else:
        shutil.copy2(src, dst)


def entry_sources(src_entry):
    fs = os.path.join(src_entry, 'finalscf')
    f51 = _first(f'{fs}/fort.51') or _first(f'{fs}/fort.51.gz')
    out = _first(f'{fs}/finalscf_*.out') or _first(f'{fs}/finalscf_*.out.gz')
    return f51, out


def convert(args):
    """Copy the small files, rebuild finalscf/mcmillan_cutoff_results.csv in GL from fort.51 + .out, regenerate
    results.json; returns (name, status, message)."""
    src_entry, dst_base, cutoff_mode, dos_norm, finish = args
    name = os.path.basename(src_entry.rstrip('/')); dst_entry = os.path.join(dst_base, name); dfs = os.path.join(dst_entry, 'finalscf')
    try:
        if finish:                       # fort.51 + .out were produced by the SLURM rerun directly in DST
            f51, out = entry_sources(dst_entry)
            if not (f51 and out): return name, 'pending', 'rerun output not there yet'
        else:
            f51, out = entry_sources(src_entry)
            if not (f51 and out): return name, 'missing', 'no fort.51 / finalscf_*.out in SRC'
            os.makedirs(dfs, exist_ok=True)
            for rel in KEEP:
                os.makedirs(os.path.dirname(os.path.join(dst_entry, rel)), exist_ok=True)
                shutil.copy2(os.path.join(src_entry, rel), os.path.join(dst_entry, rel))
            _copy_plain(f51, os.path.join(dfs, 'fort.51'))
            _copy_plain(out, os.path.join(dfs, os.path.basename(out)))
        if not os.path.exists(os.path.join(dfs, 'fort.51')):
            _copy_plain(f51, os.path.join(dfs, 'fort.51'))
        rp = json.load(open(os.path.join(dst_entry, 'run_params.json')))
        ds = json.load(open(os.path.join(dst_entry, 'debye', 'debye_summary.json')))
        run_mcmillan_cutoff_sweep(workdir=dfs, run_kkr=False, mixture_mass=rp['mixture_mass'], theta_d=ds.get('thetaDB_K'),
                                  cutoff_mode=cutoff_mode, dos_normalization=dos_norm, elements=rp['element_labels'])
        res = process_kkr(path=dst_base, dirname=name, allow_missing_theta=True)
        if res is None: return name, 'error', 'process_kkr returned None'
        res['eta_variant'] = f'{cutoff_mode}+{dos_norm}'
        with open(os.path.join(dst_entry, 'results.json'), 'w') as f: json.dump(res, f, indent=1, default=float)
        for b in BULKY + [os.path.basename(p) for p in glob.glob(f'{dfs}/finalscf_*') if not p.endswith('_results.csv')]:
            p = os.path.join(dfs, b)
            if os.path.exists(p): os.remove(p)
        return name, 'ok', f"eta_mix {res.get('eta_mix')}"
    except Exception as e:
        return name, 'error', repr(e)


def rerun_row(src_entry, name):
    rj = json.load(open(os.path.join(src_entry, 'results.json'))); rp = json.load(open(os.path.join(src_entry, 'run_params.json')))
    kp = rj.get('KKR_PARAMS_FINALSCF') or rp.get('KKR_PARAMS_FINALSCF') or {}
    conc = [float(c) for c in rp['concentrations']]; s = sum(conc); conc = [c / s for c in conc]
    return dict(name=name, elements=' '.join(str(ELEMENTS[e].atomic_number) for e in rp['element_labels']),
                concentrations=' '.join(f'{c:.6f}' for c in conc), a0=rj['a0_bohr'], ew=kp.get('ew', 0.6), xc=kp.get('xc', 'pbe'),
                rel=kp.get('rel', 'sra'), bzqlty=kp.get('bzqlty', 20), mxl=kp.get('mxl', 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--elements', default=None, help='comma list; keep entries whose non-zero components are all in it')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--cutoff-mode', default='gg'); ap.add_argument('--dos-normalization', default='lloyd')
    ap.add_argument('--finish', action='store_true', help='convert entries of DST/rerun_finalscf.csv that now have fort.51 + .out in DST')
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst); os.makedirs(dst, exist_ok=True)
    subset = set(a.elements.split(',')) if a.elements else None
    t0 = time.time()
    if a.finish:
        names = [r['name'] for r in csv.DictReader(open(os.path.join(dst, 'rerun_finalscf.csv')))]
        names = [n for n in names if not os.path.exists(os.path.join(dst, n, 'results.json'))]
        jobs = [(os.path.join(src, n), dst, a.cutoff_mode, a.dos_normalization, True) for n in names]
    else:
        for p in glob.glob(os.path.join(src, 'parameters*.json')): shutil.copy2(p, dst)
        jobs, skipped_el, missing = [], 0, []
        for n in sorted(os.listdir(src)):
            e = os.path.join(src, n)
            if not (os.path.isfile(os.path.join(e, 'results.json')) and os.path.isfile(os.path.join(e, 'run_params.json'))): continue
            rp = json.load(open(os.path.join(e, 'run_params.json')))
            if subset is not None and any(float(c) > 0 and l not in subset for l, c in zip(rp['element_labels'], rp['concentrations'])):
                skipped_el += 1; continue
            if not os.path.isfile(os.path.join(e, 'debye', 'debye_summary.json')): missing.append((n, 'no debye_summary.json')); continue
            f51, out = entry_sources(e)
            if not (f51 and out): missing.append((n, 'rerun')); continue
            jobs.append((e, dst, a.cutoff_mode, a.dos_normalization, False))
        rer = [rerun_row(os.path.join(src, n), n) for n, why in missing if why == 'rerun']
        if rer:
            with open(os.path.join(dst, 'rerun_finalscf.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rer[0])); w.writeheader(); w.writerows(rer)
        print(f'{len(jobs)} entries to convert, {skipped_el} skipped by --elements, {len(rer)} need a finalscf rerun (DST/rerun_finalscf.csv), '
              f'{len(missing) - len(rer)} without debye_summary.json')
    with Pool(a.workers) as p: res = p.map(convert, jobs)
    counts = {}
    for n, st, msg in res:
        counts[st] = counts.get(st, 0) + 1
        if st not in ('ok',): print(f'  {st:8s} {n}: {msg}')
    print(f'done in {time.time() - t0:.0f} s: {counts}')
    git = subprocess.run(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    info_p = os.path.join(dst, 'slim_info.json'); info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    info['sources'] = sorted(set(info.get('sources', []) + [src]))
    info.update(src=src, cutoff_mode=a.cutoff_mode, dos_normalization=a.dos_normalization, elements=a.elements, git=git,
                date=time.strftime('%Y-%m-%d %H:%M'), n_entries=sum(os.path.isfile(os.path.join(dst, n, 'results.json')) for n in os.listdir(dst)),
                **({'last_finish_counts': counts} if a.finish else {'counts': counts}))
    json.dump(info, open(info_p, 'w'), indent=1)


if __name__ == '__main__':
    main()
