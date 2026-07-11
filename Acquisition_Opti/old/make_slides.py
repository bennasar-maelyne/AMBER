"""
Generate meeting slides for DWI acquisition optimisation.
Style: plain PowerPoint, identical to Meeting_03-25.pptx
Run: python make_slides.py
Output: Figures/meeting_slides.pptx
"""

import os
import io
import fitz
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_OPT  = os.path.join(THIS_DIR, "Figures", "opt_results")
FIG_T2   = os.path.join(THIS_DIR, "Figures", "t2_sensitivity")
OUT_PATH = os.path.join(THIS_DIR, "Figures", "meeting_slides.pptx")

# Use the reference file as template so fonts/theme are identical
TEMPLATE = os.path.join(THIS_DIR, "..", "Paper", "Meeting_03-25.pptx")

prs = Presentation(TEMPLATE)
# Remove all existing slides using the xml slide list
from pptx.oxml.ns import qn
from lxml import etree
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    rId = sldId.get(qn('r:id'))
    if rId:
        try:
            prs.part.drop_rel(rId)
        except Exception:
            pass
    sldIdLst.remove(sldId)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height

# Find layouts by name
layouts = {lay.name: lay for lay in prs.slide_layouts}
LAY_TITLE   = layouts.get("Diapositive de titre") or prs.slide_layouts[0]
LAY_CONTENT = layouts.get("Titre et contenu")     or prs.slide_layouts[1]
LAY_BLANK   = layouts.get("Vide")                 or prs.slide_layouts[6]


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_slide(layout):
    slide = prs.slides.add_slide(layout)
    # Remove all placeholder shapes that came with the layout
    # (we add our own content)
    return slide


def set_title(slide, text):
    """Set title via placeholder if available, else add textbox."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.text = text
            return
    # fallback
    txb = slide.shapes.add_textbox(Inches(0.92), Inches(0.4), Inches(11.5), Inches(0.9))
    txb.text_frame.paragraphs[0].text = text


def add_text(slide, text, l, t, w, h, bold=False, size=None, align=PP_ALIGN.LEFT):
    """Plain textbox, no fill, no border, inherits theme style."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf  = box.text_frame
    tf.word_wrap = True
    box.fill.background()
    box.line.fill.background()
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    if bold:
        run.font.bold = True
    if size:
        run.font.size = size
    return box


def add_bullets(slide, lines, l, t, w, h, size=None):
    """Multi-line textbox, plain style."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf  = box.text_frame
    tf.word_wrap = True
    box.fill.background()
    box.line.fill.background()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(2)
        run = p.add_run()
        run.text = line
        if size:
            run.font.size = size
    return box


def add_figure(slide, pdf_path, l, t, w, h):
    if not os.path.exists(pdf_path):
        add_text(slide, "[Figure: " + os.path.basename(pdf_path) + "]", l, t, w, h)
        return
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(180 / 72, 180 / 72)
    pix = doc[0].get_pixmap(matrix=mat)
    slide.shapes.add_picture(io.BytesIO(pix.tobytes("png")), l, t, w, h)


# ==============================================================================
# Slide 1 — Titre
# ==============================================================================
slide = add_slide(LAY_TITLE)
set_title(slide, "DWI Acquisition Optimisation")
for ph in slide.placeholders:
    if ph.placeholder_format.idx == 1:
        ph.text = "Directions, preliminary results and open questions\nMaelyne Bennasar  -  May 2026"
print("Slide 1 done")


# ==============================================================================
# Slide 2 — Contexte
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Context")

add_bullets(slide, [
    "Validation (done): LUT validated on 2 patients, simulated signals consistent with in vivo DWI",
    "",
    "Next step (this meeting): use the LUT to find the best (b-value, TD) to discriminate tissue types",
    "",
    "Two levels of modelling:",
    "  L1 - diffusion only",
    "  L2 - + T2 attenuation + thermal noise  (TE = TD + delta)",
    "",
    "Pipeline:",
    "  Tumour phenotypes  ->  LHS sampling  ->  LUT signals  ->  CNR / AUC / Cohen d  ->  (b*, TD*) optimal",
], Inches(0.92), Inches(1.6), Inches(11.5), Inches(4.5))

add_text(slide, "Q: Is there a universal acquisition parameter, or does it depend on which tissue types to discriminate?",
         Inches(0.92), Inches(6.3), Inches(11.5), Inches(0.8), bold=True)
print("Slide 2 done")


# ==============================================================================
# Slide 3 — Phenotype catalogue
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Tumour phenotype catalogue")

headers = ["Phenotype", "f", "Dex (um2/ms)", "rmean (um)", "rsd (um)", "Distinctive feature"]
rows_data = [
    ["Edema",       "0.05 - 0.15",  "2.0 - 3.0", "8 - 15",  "1 - 5",   "High Dex"],
    ["Cyst",        "0.005 - 0.05", "2.5 - 3.0", "8 - 15",  "0 - 2",   "Very low f"],
    ["Large cells", "0.35 - 0.60",  "1.5 - 2.5", "12 - 20", "2 - 6",   "High rmean"],
    ["Small cells", "0.50 - 0.75",  "0.8 - 1.5", "3 - 7",   "0.5 - 2", "Low rmean, high f"],
    ["Fibrosis",    "0.40 - 0.65",  "0.5 - 1.0", "5 - 9",   "1 - 3",   "Low Dex"],
    ["Necrosis",    "-",            "-",          "-",       "-",       "Deferred (2-population LUT unavailable)"],
]

n_rows = 1 + len(rows_data)
n_cols = len(headers)
tbl = slide.shapes.add_table(
    n_rows, n_cols,
    Inches(0.5), Inches(1.55),
    Inches(12.33), Inches(4.5)
).table

# Column widths
col_widths = [Inches(1.6), Inches(1.4), Inches(1.6), Inches(1.5), Inches(1.3), Inches(4.93)]
for i, w in enumerate(col_widths):
    tbl.columns[i].width = w

# Header row
for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = h
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.runs[0]
    run.font.bold = True

# Data rows
for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        cell = tbl.cell(i + 1, j)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if j < 5 else PP_ALIGN.LEFT

add_text(slide,
    "! All bounds marked VALIDATE - not confirmed with literature yet.",
    Inches(0.5), Inches(6.2), Inches(11.5), Inches(0.45))

add_text(slide, "Q: Are these bounds realistic? Do the phenotypes overlap in parameter space?",
         Inches(0.5), Inches(6.65), Inches(11.5), Inches(0.55), bold=True)
print("Slide 3 done")


# ==============================================================================
# Slide 4 — Method L1
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Method - Level 1 (diffusion only)")

add_bullets(slide, [
    "For each pair of phenotypes:",
    "  - 500 Latin Hypercube samples per phenotype",
    "  - Delaunay interpolation on the LUT -> signals S/S0(b, TD)",
    "  - Metrics computed at each grid point (b, TD):",
    "      CNR = |mean_A - mean_B| / (std_A + std_B)",
    "      Cohen d = |mean_A - mean_B| / std_pooled",
    "      AUC (ROC curve)",
    "",
    "Grid: b in {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5} ms/um2  (10 pts)",
    "      TD in {10, 20, 30, 40, 50, 60, 70, 80} ms  (8 pts)",
    "",
    "-> 10 pairs x 80 grid points = 800 CNR values computed",
], Inches(0.92), Inches(1.6), Inches(11.5), Inches(4.5))

add_text(slide, "Q: Is the grid dense enough? The true optimum may fall between grid points.",
         Inches(0.92), Inches(6.3), Inches(11.5), Inches(0.8), bold=True)
print("Slide 4 done")


# ==============================================================================
# Slide 5 — Signal decay
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Level 1 - Signal decay per phenotype")

add_figure(slide,
           os.path.join(FIG_OPT, "fig1_signal_decay.pdf"),
           Inches(0.5), Inches(1.5), Inches(8.5), Inches(5.0))

add_bullets(slide, [
    "Cyst: high signal at all b",
    "(very low f -> little restriction)",
    "",
    "Fibrosis: fast decay",
    "(low Dex -> strong restriction)",
    "",
    "Small cells: low signal,",
    "small std",
    "",
    "! Small std reflects narrow",
    "  LHS bounds, not necessarily",
    "  true biological variability",
], Inches(9.2), Inches(1.6), Inches(3.9), Inches(4.8))

add_text(slide, "Q: Does the std reflect real biological variability or just the width of the parameter bounds?",
         Inches(0.5), Inches(6.55), Inches(12.0), Inches(0.6), bold=True)
print("Slide 5 done")


# ==============================================================================
# Slide 6 — CNR heatmaps
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Level 1 - CNR on the (b, TD) grid")

add_figure(slide,
           os.path.join(FIG_OPT, "fig2_cnr_heatmaps.pdf"),
           Inches(0.5), Inches(1.5), Inches(9.5), Inches(5.0))

add_bullets(slide, [
    "TD = 80 ms optimal",
    "for all pairs",
    "-> always the grid max",
    "",
    "b* varies per pair",
    "",
    "CNR max = 12.5",
    "(Cyst / Fibrosis)",
    "CNR min = 1.4",
    "(Small / Fibrosis)",
], Inches(10.2), Inches(1.6), Inches(2.9), Inches(4.5))

add_text(slide, "Q: TD = 80 ms wins because it is the grid maximum - what happens at TD = 150 ms or more?",
         Inches(0.5), Inches(6.55), Inches(12.0), Inches(0.6), bold=True)
print("Slide 6 done")


# ==============================================================================
# Slide 7 — Summary table + AUC problem
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Level 1 - Summary table and AUC issue")

add_figure(slide,
           os.path.join(FIG_OPT, "fig4_summary_table.pdf"),
           Inches(0.5), Inches(1.5), Inches(8.0), Inches(3.2))

add_bullets(slide, [
    "AUC = 1.000 for 9/10 pairs",
    "-> distributions perfectly separated",
    "",
    "This is a consequence of the",
    "phenotype definition: bounds do",
    "not overlap in parameter space.",
    "Not a meaningful result.",
    "",
    "CNR remains informative",
    "(quantifies the separation margin).",
    "",
    "For a realistic test: define",
    "overlapping phenotypes.",
], Inches(8.7), Inches(1.5), Inches(4.4), Inches(4.8))

add_text(slide, "Q: How to define 'hard-to-discriminate' phenotypes for a clinically relevant optimisation?",
         Inches(0.5), Inches(6.55), Inches(12.0), Inches(0.6), bold=True)
print("Slide 7 done")


# ==============================================================================
# Slide 8 — Level 2
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Level 2 - T2 attenuation + thermal noise")

add_bullets(slide, [
    "Physical model (PGSE standard sequence, TE = TD + delta):",
    "  S_measured(b, TD) = S_LUT(b, TD) x exp(-TE / T2) + Rician noise",
    "  sigma_noise = 1 / SNR0     SNR0 = 40 (25th percentile from noise maps, 2 patients)",
    "",
    "T2 assumption - critical issue:",
    "  T2 = 100 ms assumed (glioma at 3T, literature value)",
    "  Not measurable from our data: S0(TD=19ms) / S0(TD=49ms) ~ 1.00-1.01",
    "  T2=100ms would predict a ratio of 1.35 -> TE is likely fixed in the acquisition protocol",
    "  -> Level 2 models a hypothetical protocol, not the actual scanner setup",
    "",
    "Consequence on the optimum:",
    "  L1: TD* = 80 ms (diffusion dominates)",
    "  L2: TD* = 20 ms (T2 penalises long TD)    CNR drops ~40-60%",
], Inches(0.92), Inches(1.6), Inches(11.5), Inches(4.8))

add_text(slide, "Q: Does the scanner use a fixed TE or TE = TD + delta? This determines which level applies.",
         Inches(0.92), Inches(6.4), Inches(11.5), Inches(0.7), bold=True)
print("Slide 8 done")


# ==============================================================================
# Slide 9 — L1 vs L2
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Level 1 vs Level 2 - Comparison of optima")

add_figure(slide,
           os.path.join(FIG_OPT, "fig5_cnr_l1_vs_l2.pdf"),
           Inches(0.5), Inches(1.5), Inches(8.7), Inches(5.1))

add_bullets(slide, [
    "L1 -> L2: TD* 80 -> 20 ms",
    "(all pairs)",
    "",
    "b* also shifts:",
    "Cyst/Fibrosis:",
    "  b* 0.25 -> 0.50 ms/um2",
    "",
    "CNR / ~1.7 on average",
    "",
    "Small/Fibrosis remains",
    "hardest pair (CNR < 1 in L2)",
], Inches(9.4), Inches(1.6), Inches(3.7), Inches(5.0))

add_text(slide, "Q: Is TD = 20 ms clinically feasible? What is the minimum accessible diffusion time?",
         Inches(0.5), Inches(6.65), Inches(12.0), Inches(0.55), bold=True)
print("Slide 9 done")


# ==============================================================================
# Slide 10 — T2 sensitivity
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "T2 sensitivity analysis")

add_figure(slide,
           os.path.join(FIG_T2, "fig_t2_optimum.pdf"),
           Inches(0.5), Inches(1.5), Inches(9.0), Inches(4.7))

add_bullets(slide, [
    "T2 swept from 50 to 500 ms",
    "SNR0 = 40 fixed",
    "",
    "Low T2 (50 ms):",
    "  very short TD*, low CNR",
    "",
    "High T2 (500 ms):",
    "  T2 penalty negligible,",
    "  TD* converges to L1 result",
    "",
    "Critical transition ~100-150 ms",
    "",
    "! T2 = 100 ms assumed but",
    "  not verified on our data",
], Inches(9.7), Inches(1.6), Inches(3.4), Inches(4.8))

add_text(slide, "Q: Should T2 be measured in vivo per patient, or is a literature value acceptable for the paper?",
         Inches(0.5), Inches(6.4), Inches(12.0), Inches(0.7), bold=True)
print("Slide 10 done")


# ==============================================================================
# Slide 11 — Open issues
# ==============================================================================
slide = add_slide(LAY_CONTENT)
set_title(slide, "Open issues")

add_bullets(slide, [
    "Phenotype bounds not validated",
    "  All bounds marked VALIDATE in the code - need literature confirmation",
    "",
    "AUC = 1 not informative",
    "  Phenotype ranges do not overlap -> too easy to discriminate, not a realistic test",
    "",
    "Discrete grid",
    "  TD max = 80 ms in the current LUT -> true optimum may be outside the grid",
    "",
    "T2 not measurable from our data",
    "  TE appears fixed in the current protocol -> Level 2 is a hypothetical scenario",
    "",
    "Necrosis missing",
    "  Two-population LUT not available -> clinically important tissue type",
    "",
    "SNR0 calibrated on 2 patients at TD = 19 ms only - inter-patient variability not modelled",
], Inches(0.92), Inches(1.6), Inches(11.5), Inches(5.0))

add_text(slide, "Q: Which of these issues are blocking for publication, and which can be addressed with a disclaimer?",
         Inches(0.92), Inches(6.7), Inches(11.5), Inches(0.55), bold=True)
print("Slide 11 done")


# ── Save ───────────────────────────────────────────────────────────────────────
prs.save(OUT_PATH)
print("Saved ->", OUT_PATH, " |", len(prs.slides), "slides")
