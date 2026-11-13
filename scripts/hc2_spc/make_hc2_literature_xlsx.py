"""
Build results/HC2_literature.xlsx — experimental upper critical fields, same layout idea as results/Tc_literature.xlsx.

Input: results/HC2_literature_entries.json (list of entries collected from the papers cited in Tc_literature.xlsx;
each entry has "row" = "rowN" (row of Tc_literature.xlsx), or "new" with "composition_paper" + "elements" {el: at.%}).
Composition, Link and Status are copied from Tc_literature.xlsx for "rowN" entries, so both sheets share the alloy names.
Literature only — no KKR numbers in this file.

    python scripts/hc2_spc/make_hc2_literature_xlsx.py
"""
import json

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

TC_XLSX = "results/Tc_literature.xlsx"
ENTRIES = "results/HC2_literature_entries.json"
OUT = "results/HC2_literature.xlsx"
EL = ["Sc", "Ti", "Y", "Zr", "Nb", "Mo", "Ru", "La", "Hf", "Ta", "W", "Re", "V"]      # Tc_literature cols B..N
COL_LINK, COL_TC, COL_AUTHOR, COL_STATUS = 15, 19, 26, 31

EXP = [("Tc [K]", "Tc_K", 7), ("μ0Hc2(0) [T]", "Hc2_0_T", 9), ("Hc2 method", "Hc2_method", 28),
       ("Hc2 criterion", "Hc2_criterion", 24), ("|μ0dHc2/dT|Tc [T/K]", "dHc2dT_T_per_K", 10),
       ("ρ0 [μΩcm]", "rho0_uOhm_cm", 8), ("γ [mJ/mol K²]", "gamma_mJ_molK2", 9), ("α_M", "alpha_M", 6),
       ("ξ_GL [nm]", "xi_GL_nm", 7), ("Hc2 alt (method)", "Hc2_alt", 34)]
META = [("First Author", "first_author", 14), ("Value source", "value_source", 10), ("Where", "where", 30),
        ("Quote", "quote", 40), ("Status (Tc_literature)", "status", 22), ("Tc_literature row", "tc_row", 9),
        ("Tc_literature Tc [K]", "tc_lit", 8), ("Notes", "notes", 60)]

FILL = {"TAB": "FFE2EFDA", "FIG-LBL": "FFE2EFDA", "FIG-EST": "FFFFF2CC", "ABS": "FFFFF2CC", "SEC": "FFFCE4D6"}


def main():
    src = openpyxl.load_workbook(TC_XLSX, data_only=True)["Tc_literature"]
    entries = json.load(open(ENTRIES))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Hc2_literature"
    c0 = 1 + len(EL) + 2                                      # first experimental column (after name, elements, Link)
    m0 = c0 + len(EXP)
    ws.cell(1, 1, "Composition")
    ws.cell(1, c0, "Experimental (from the cited paper)")
    ws.cell(1, m0, "Metadata")
    ws.cell(2, 1, "Name (as in Tc_literature.xlsx)")
    for i, e in enumerate(EL):
        ws.cell(2, 2 + i, e)
    ws.cell(2, 2 + len(EL), "Link")
    for j, (h, _, w) in enumerate(EXP + META):
        ws.cell(2, c0 + j, h)
        ws.column_dimensions[get_column_letter(c0 + j)].width = w
    ws.column_dimensions["A"].width = 26
    for i in range(len(EL)):
        ws.column_dimensions[get_column_letter(2 + i)].width = 5.5
    ws.column_dimensions[get_column_letter(2 + len(EL))].width = 30
    for c in range(1, m0 + len(META)):
        for r in (1, 2):
            ws.cell(r, c).font = Font(bold=True)
            ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")

    r = 3
    for e in entries:
        if e.get("Hc2_0_T") is None and not e.get("dHc2dT_T_per_K"):
            continue
        row = e.get("row", "")
        if row.startswith("row"):
            sr = int(row[3:])
            ws.cell(r, 1, src.cell(sr, 1).value)
            for i in range(len(EL)):
                v = src.cell(sr, 2 + i).value
                if isinstance(v, (int, float)):
                    ws.cell(r, 2 + i, v)
            ws.cell(r, 2 + len(EL), e.get("doi") or src.cell(sr, COL_LINK).value)   # paper of the Hc2 value
            e = dict(e, status=src.cell(sr, COL_STATUS).value, tc_row=sr, tc_lit=src.cell(sr, COL_TC).value)
        else:                                                  # sample not in Tc_literature.xlsx
            ws.cell(r, 1, e.get("name"))
            for el, v in (e.get("elements") or {}).items():
                ws.cell(r, 2 + EL.index(el), v)
            if e.get("comp_row"):                              # e.g. annealed variant: nominal composition of that row
                for i in range(len(EL)):
                    v = src.cell(e["comp_row"], 2 + i).value
                    if isinstance(v, (int, float)):
                        ws.cell(r, 2 + i, v)
            ws.cell(r, 2 + len(EL), e.get("doi"))
            e = dict(e, status="not in Tc_literature", tc_row=None, tc_lit=None)
        if isinstance(e.get("dHc2dT_T_per_K"), (int, float)):
            e = dict(e, dHc2dT_T_per_K=abs(e["dHc2dT_T_per_K"]))
        for j, (_, k, _) in enumerate(EXP + META):
            v = e.get(k)
            if v is not None and v != "":
                ws.cell(r, c0 + j, v)
        fill = FILL.get(str(e.get("value_source", "")).split()[0] if e.get("value_source") else "", None)
        if fill:
            ws.cell(r, 1).fill = PatternFill("solid", fgColor=fill)
        ws.cell(r, c0 + 1).font = Font(bold=True)
        r += 1
    ws.freeze_panes = "B3"

    lg = wb.create_sheet("Legend")
    for row in [
        ("Purpose", "Literature-only experimental upper critical fields for the alloys of Tc_literature.xlsx "
                    "(validation of the KKR/WHH Hc2 estimate, branch rtop/hc2). No KKR results here."),
        ("Rows", "One row per (sample, paper). Name, composition (at.%) and Status copied from Tc_literature.xlsx; Link = paper of the Hc2 value "
                 "(column 'Tc_literature row'); rows with status 'not in Tc_literature' are extra samples from the same papers."),
        ("μ0Hc2(0)", "Primary value as given by the authors; HOW it was obtained is in 'Hc2 method' — full WHH (α_M, λ_SO) "
                     "fits give the lowest values; GL (1−t²) and power-law / linear extrapolations are systematically higher. "
                     "Other values from the same paper in 'Hc2 alt'."),
        ("|μ0dHc2/dT|", "Initial slope at Tc, only if reported by the paper (never computed by us)."),
        ("Value source", "TAB = table/text of the primary paper (green); FIG-LBL = number printed on a figure (green); "
                         "FIG-EST = read by eye from a figure (yellow); ABS = abstract only (yellow); SEC = from a review / "
                         "other paper citing it (orange)."),
        ("Status", "Copied from Tc_literature.xlsx (baseline / candidate / caution …)."),
        ("Not found", "Sheet 'checked_no_Hc2' lists papers that were checked and report no Hc2 or could not be accessed."),
        ("Built", "scripts/hc2_spc/make_hc2_literature_xlsx.py from results/HC2_literature_entries.json "
                  "(collected 2026-11-13: local PDFs in papers_data/ + web, by Claude with sub-agents)."),
    ]:
        lg.append(row)
    lg.column_dimensions["A"].width = 16
    lg.column_dimensions["B"].width = 140
    for c in lg["A"]:
        c.font = Font(bold=True)

    nf = wb.create_sheet("checked_no_Hc2")
    nf.append(["First Author", "Link", "Tc_literature rows", "Result", "Notes"])
    for e in entries:
        if e.get("Hc2_0_T") is None and not e.get("dHc2dT_T_per_K"):
            nf.append([e.get("first_author"), e.get("doi"), e.get("row"), e.get("value_source"), e.get("notes")])
    for col, w in zip("ABCDE", (14, 34, 20, 12, 100)):
        nf.column_dimensions[col].width = w
    wb.save(OUT)
    print(f"{OUT}: {r - 3} Hc2 rows, {nf.max_row - 1} checked without Hc2")


if __name__ == "__main__":
    main()
