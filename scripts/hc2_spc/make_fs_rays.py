"""
Fermi-surface sampling for the Hc2 estimate: radial rays from Gamma over the irreducible wedge of the bcc BZ.

AkaiKKR (spmain.f: go*3) always reads spc k-points in format (c) = a polyline through at most 25 points with a
total number of k-points spread equidistantly. Rays are therefore packed as
    G -> A1 -> A2 -> G -> A3 -> A4 -> G ...      (16 rays per run; the short A1->A2 boundary pieces are unused)
Directions: Fibonacci (equal-area) points on the unit sphere restricted to the wedge nx >= ny >= nz >= 0
(1/48 of the sphere, O_h) -> equal solid-angle weights. Ray end A = n / (nx + ny) in units 2pi/a
(bcc BZ = rhombic dodecahedron, faces |ki| + |kj| = 1).

Writes into <sysdir>/fs/ :  fs_XX.inp (spc on the potential of <sysdir>), fs_rays.json (vectors, ray ids), run_fs.sh
Usage: python scripts/hc2_spc/make_fs_rays.py hc2_spc/Ta34Nb33Hf8Zr14Ti11 [--n_sphere 2400] [--dk 0.0032]
"""
import argparse
import json
import os
import re

import numpy as np

RAYS_PER_RUN = 16


def wedge_directions(n_sphere):
    i = np.arange(n_sphere) + 0.5
    z = 1 - 2 * i / n_sphere
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    v = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    m = (v[:, 0] >= v[:, 1]) & (v[:, 1] >= v[:, 2]) & (v[:, 2] >= 0)
    return v[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sysdir")
    ap.add_argument("--n_sphere", type=int, default=2400, help="points on the full sphere (~n/48 rays in the wedge)")
    ap.add_argument("--dk", type=float, default=0.0032, help="target k spacing along the path [2pi/a]")
    ap.add_argument("--edelt", type=float, default=None, help="override edelt (imag. energy) of the spc runs [Ry]")
    ap.add_argument("--sub", default="fs", help="output subdirectory of <sysdir>")
    args = ap.parse_args()

    spc_ef = open(os.path.join(args.sysdir, "spc_fine.inp")).read()
    head = spc_ef.split("# k-points")[0]                      # spc block of the fine run (ewidth 0.05, same pot)
    pot = re.search(r"^spc\S*\s+(\S+)", head, re.M).group(1)
    m = re.search(r"^(\s+)(\S+)(\s+\S+\s+\S+\s+\S+\s+\S+\s+2nd)", head, re.M)   # edelt ewidth reltyp sdftyp magtyp record
    edelt = float(m.group(2))
    if args.edelt is not None:
        head = head[:m.start()] + m.group(1) + f"{args.edelt}" + m.group(3) + head[m.end():]
        edelt = args.edelt

    dirs = wedge_directions(args.n_sphere)
    ends = dirs / (dirs[:, 0] + dirs[:, 1])[:, None]
    fsdir = os.path.join(args.sysdir, args.sub)
    os.makedirs(fsdir, exist_ok=True)

    runs = []
    for r0 in range(0, len(dirs), RAYS_PER_RUN):
        idx = list(range(r0, min(r0 + RAYS_PER_RUN, len(dirs))))
        pts, rays = [np.zeros(3)], []
        for j in range(0, len(idx), 2):
            pair = idx[j:j + 2]
            for q, ri in enumerate(pair):
                pts.append(ends[ri])
                # segment index (between pts[-2] and pts[-1]); first of pair is G->A, second is A->G
                if q == 0:
                    rays.append(dict(ray=ri, seg=len(pts) - 2, outward=True))
            if len(pair) == 2:
                pts.append(np.zeros(3))
                rays.append(dict(ray=pair[1], seg=len(pts) - 2, outward=False))
            else:
                pts.append(np.zeros(3))                       # odd ray: back to G (segment unused)
        pts = np.array(pts)
        L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        nk = int(np.ceil(L / args.dk)) + 1
        name = f"fs_{len(runs):02d}"
        with open(os.path.join(fsdir, name + ".inp"), "w") as f:
            f.write(head.replace(f" {pot}\n", f" ../{pot}\n", 1))
            f.write("# k-points for spc: total number, then path points (units 2pi/a)\n")
            f.write(f" {nk}\n")
            for p in pts:
                f.write(f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
            f.write("end\n")
        runs.append(dict(name=name, nk=nk, points=pts.tolist(), rays=rays))

    json.dump(dict(sysdir=args.sysdir, edelt_spc=edelt, n_sphere=args.n_sphere, n_rays=len(dirs), dirs=dirs.tolist(),
                   ends=ends.tolist(), domega=4 * np.pi / args.n_sphere, runs=runs),
              open(os.path.join(fsdir, "fs_rays.json"), "w"), indent=1)
    with open(os.path.join(fsdir, "run_fs.sh"), "w") as f:
        f.write("#!/bin/bash\n# A(k,E) on radial rays (Fermi-surface average for Hc2); needs the potential from ../run.sh\n")
        f.write('set -e\ncd "$(dirname "$0")"\n')
        f.write('BIN=${AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}\n')
        f.write(f'[ -f ../{pot} ] || {{ echo "missing ../{pot}: run ../run.sh first"; exit 1; }}\n')
        for r in runs:
            f.write(f"$BIN < {r['name']}.inp > {r['name']}.out && mv ../{pot}_up.spc {r['name']}_up.spc\n")
        f.write(f"rm -f ../{pot}_dn.spc\n")
    os.chmod(os.path.join(fsdir, "run_fs.sh"), 0o755)
    print(f"{fsdir}: {len(dirs)} rays, {len(runs)} runs, nk per run {min(r['nk'] for r in runs)}-{max(r['nk'] for r in runs)}")


if __name__ == "__main__":
    main()
