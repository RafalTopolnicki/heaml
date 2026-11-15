"""Matplotlib style shared by the paper figures (PRB: single column 3.4 in, double column 7.0 in)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'lines.linewidth': 1.0, 'axes.linewidth': 0.6, 'savefig.dpi': 300, 'figure.dpi': 120,
    'legend.frameon': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'mathtext.fontset': 'stix',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
})
SINGLE, DOUBLE = 3.4, 7.0

# one colour per element, identical in every figure
ALL_ELEMENTS = ['Nb', 'Zr', 'Ti', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru', 'V', 'Sc']
_pal = plt.cm.tab20.colors
EL_COLORS = {'Nb': _pal[0], 'Zr': _pal[2], 'Ti': _pal[4], 'Hf': _pal[6], 'Ta': _pal[8], 'Mo': _pal[10],
             'W': _pal[12], 'Y': _pal[14], 'Re': _pal[16], 'Ru': _pal[18], 'V': _pal[1], 'Sc': _pal[3]}
C_A, C_B = '#1f77b4', '#d62728'
C_FIT = '#2ca02c'   # linear model of Fig. 1 (fit line there, rescaled front in Fig. 2)


def save(fig, name):
    from common import FIG
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    print('saved', name)
