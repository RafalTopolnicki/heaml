"""Shared helpers for the eta/RMTA tests: fort.51 loading (gz aware) and element labels."""
import gzip, json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from macmillan import parse_fort51   # noqa: E402

_TMP = tempfile.mkdtemp(prefix='fort51_')

def fort51_path(p):
    """Accept fort.51, fort.51.gz, or a process_hea workdir (uses <dir>/finalscf/fort.51[.gz]); returns a plain path."""
    if os.path.isdir(p):
        for c in ('finalscf/fort.51', 'finalscf/fort.51.gz', 'fort.51', 'fort.51.gz'):
            if os.path.exists(os.path.join(p, c)):
                p = os.path.join(p, c); break
        else:
            raise FileNotFoundError(f'no fort.51 under {p}')
    if p.endswith('.gz'):
        out = os.path.join(_TMP, os.path.basename(os.path.dirname(os.path.dirname(p)) or 'x') + '_fort.51')
        with gzip.open(p, 'rb') as f, open(out, 'wb') as g:
            shutil.copyfileobj(f, g)
        return out
    return p

def labels_for(p):
    """Element labels from run_params.json of the workdir containing fort.51 (else Z numbers from the file order)."""
    d = p if os.path.isdir(p) else os.path.dirname(os.path.dirname(p))
    rp = os.path.join(d, 'run_params.json')
    if os.path.exists(rp):
        return json.load(open(rp))['element_labels']
    return None
