"""
Reconstruction complète de 'DWI acquisition optimisation.pptx'.
Template: Paper/Meeting_03-25.pptx (même thème/fonts).
"""

import os
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

# ── Chemins ────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))

TEMPLATE = os.path.join(ROOT, "Paper", "Meeting_03-25.pptx")
OUT_PATH = os.path.join(THIS_DIR, "Figures", "DWI acquisition optimisation.pptx")

FIGS = {
    "fig1_signal_decay":    os.path.join(THIS_DIR, "Figures", "opt_results",    "fig1_signal_decay.png"),
    "fig2_cnr_heatmaps":    os.path.join(THIS_DIR, "Figures", "opt_results",    "fig2_cnr_heatmaps.png"),
    "fig3_violin_optimal":  os.path.join(THIS_DIR, "Figures", "opt_results",    "fig3_violin_optimal.png"),
    "fig4_summary_table":   os.path.join(THIS_DIR, "Figures", "opt_results",    "fig4_summary_table.png"),
    "fig5_cnr_l1_vs_l2":    os.path.join(THIS_DIR, "Figures", "opt_results",    "fig5_cnr_l1_vs_l2.png"),
    "fig6_summary_l1_vs_l2":os.path.join(THIS_DIR, "Figures", "opt_results",    "fig6_summary_l1_vs_l2.png"),
    "fig_snr0_heatmaps":    os.path.join(THIS_DIR, "Figures", "snr_sensitivity","fig_snr0_heatmaps.png"),
    "fig_snr0_optimum":     os.path.join(THIS_DIR, "Figures", "snr_sensitivity","fig_snr0_optimum.png"),
    "fig_t2_heatmaps":      os.path.join(THIS_DIR, "Figures", "t2_sensitivity", "fig_t2_heatmaps.png"),
    "fig_t2_optimum":       os.path.join(THIS_DIR, "Figures", "t2_sensitivity", "fig_t2_optimum.png"),
    "fig_sens_all_raw":     os.path.join(THIS_DIR, "Figures", "sensitivity",    "fig_sens_all_raw.png"),
    "fig_sens_all_eff":     os.path.join(THIS_DIR, "Figures", "sensitivity",    "fig_sens_all_eff.png"),
    "fig_crlb_heatmap":     os.path.join(THIS_DIR, "Figures", "crlb",           "fig_crlb_heatmap.png"),
    "fig_crlb_vs_td":       os.path.join(THIS_DIR, "Figures", "crlb",           "fig_crlb_vs_td.png"),
    "fig_crlb_correlation": os.path.join(THIS_DIR, "Figures", "crlb",           "fig_crlb_correlation.png"),
}

# ── Load template and clear all slides ────────────────────────────────────────
prs = Presentation(TEMPLATE)
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    rId = sldId.get(qn('r:id'))
    if rId:
        try:
            prs.part.drop_rel(rId)
        except Exception:
            pass
    sldIdLst.remove(sldId)

W = prs.slide_width.inches    # 13.33"
H = prs.slide_height.inches   # 7.50"

# Find layouts
layouts = {}
for lay in prs.slide_layouts:
    try:
        layouts[lay.name] = lay
    except Exception:
        pass

LAY_TITLE   = layouts.get("Diapositive de titre", prs.slide_layouts[0])
LAY_CONTENT = layouts.get("Titre et contenu",     prs.slide_layouts[1])
LAY_BLANK   = layouts.get("Vide",                 prs.slide_layouts[6])

BLUE_DARK  = RGBColor(31, 73, 125)
BLUE_MID   = RGBColor(0, 112, 192)
RED_DARK   = RGBColor(192, 0, 0)
GREEN_DARK = RGBColor(0, 112, 0)
GREY_TEXT  = RGBColor(80, 80, 80)

print(f"Template: {len(prs.slides)} slides after clear (should be 0)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_slide(layout):
    return prs.slides.add_slide(layout)


def set_title(slide, text):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.text = text
            return
    tb = slide.shapes.add_textbox(Inches(0.92), Inches(0.40), Inches(11.5), Inches(1.0))
    tb.text_frame.paragraphs[0].text = text


def textbox(slide, lines, l, t, w, h, font_size=12, bold=False, color=None,
            align=PP_ALIGN.LEFT, line_gap_pt=3):
    """Add a transparent textbox with multiple paragraphs."""
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = box.text_frame
    tf.word_wrap = True
    box.fill.background()
    box.line.fill.background()
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(line_gap_pt)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        if color:
            run.font.color.rgb = color
    return box


def section_header(slide, text, l, t, w, h=0.40, color=BLUE_DARK):
    return textbox(slide, [text], l, t, w, h, font_size=14, bold=True, color=color)


def caption(slide, text, l, t, w, h=0.42):
    return textbox(slide, [text], l, t, w, h, font_size=9, bold=False, color=GREY_TEXT)


def add_pic_centered(slide, img_key, top_in, max_w_in, max_h_in):
    """Scale image to fit box, center horizontally. Returns actual height."""
    path = FIGS[img_key]
    img  = Image.open(path)
    iw, ih = img.size
    ratio = iw / ih
    w = min(max_w_in, max_h_in * ratio)
    h = w / ratio
    if h > max_h_in:
        h = max_h_in
        w = h * ratio
    left = (W - w) / 2
    slide.shapes.add_picture(path, Inches(left), Inches(top_in), Inches(w), Inches(h))
    return h


def add_pic(slide, img_key, l_in, t_in, w_in, h_in):
    slide.shapes.add_picture(FIGS[img_key], Inches(l_in), Inches(t_in), Inches(w_in), Inches(h_in))


def pic_dims(img_key, max_w_in, max_h_in):
    """Return (w, h) in inches, aspect-ratio preserved, fitting in box."""
    img = Image.open(FIGS[img_key])
    iw, ih = img.size
    ratio = iw / ih
    w = min(max_w_in, max_h_in * ratio)
    h = w / ratio
    if h > max_h_in:
        h = max_h_in
        w = h * ratio
    return w, h


# ==============================================================================
# Slide 1 — Titre
# ==============================================================================
slide = new_slide(LAY_TITLE)
set_title(slide, "DWI acquisition optimisation")
for ph in slide.placeholders:
    try:
        if ph.placeholder_format.idx == 1:
            ph.text = "Results and T₂ model update\nMaëlyne Bennasar  —  June 2026"
    except Exception:
        pass
print("Slide 1 done")


# ==============================================================================
# Slide 2 — Table des matières
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "Contents")
textbox(slide, [
    "1.  Phenotypes",
    "2.  Model and formulas",
    "       Signal T₂-independence  |  Noise model  |  Level 1 & Level 2",
    "       Response to Josh's question on T₂",
    "3.  L1: CNR without noise correction",
    "4.  Method with noise correction:",
    "       A. CNR discrimination — L1 vs L2  +  signal distributions",
    "       B. Sensitivity maps  (raw and noise-normalised)",
    "       C. CRLB study  (single-b, multi-b, FIM correlation)",
    "5.  T₂ sensitivity study",
    "6.  SNR₀ sensitivity study",
], l=1.0, t=1.85, w=11.3, h=5.2, font_size=13, line_gap_pt=5)
print("Slide 2 done")


# ==============================================================================
# Slide 3 — Phénotypes
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "1. Phenotypes")

from pptx.enum.text import PP_ALIGN

headers = ["Phenotype", "f", "Dex  (µm²/ms)", "rmean  (µm)", "rsd  (µm)", "Defining feature"]
rows_data = [
    ["Edema",       "0.05 – 0.15",  "2.0 – 3.0",  "8 – 15",  "1 – 5",   "High Dex (oedematous)"],
    ["Cyst",        "0.005 – 0.05", "2.5 – 3.0",  "8 – 15",  "0 – 2",   "Very low f"],
    ["Large cells", "0.35 – 0.60",  "1.5 – 2.5",  "12 – 20", "2 – 6",   "High rmean"],
    ["Small cells", "0.50 – 0.75",  "0.8 – 1.5",  "3 – 7",   "0.5 – 2", "Low rmean, high f"],
    ["Fibrosis",    "0.40 – 0.65",  "0.5 – 1.0",  "5 – 9",   "1 – 3",   "Low Dex"],
    ["Necrosis",    "—",            "—",          "—",       "—",       "Deferred (2-population LUT not yet available)"],
]

tbl = slide.shapes.add_table(
    1 + len(rows_data), len(headers),
    Inches(0.50), Inches(1.75), Inches(12.33), Inches(4.55)
).table

col_widths = [Inches(1.55), Inches(1.35), Inches(1.65), Inches(1.45), Inches(1.35), Inches(4.98)]
for i, cw in enumerate(col_widths):
    tbl.columns[i].width = cw

for j, hdr in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = hdr
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.runs[0]
    r.font.bold = True
    r.font.size = Pt(11)

for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        cell = tbl.cell(i + 1, j)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if j < 5 else PP_ALIGN.LEFT
        r = p.runs[0]
        r.font.size = Pt(11)

textbox(slide,
    "⚠  All parameter bounds are to be validated with Josh — marked VALIDATE in tumor_catalog.py",
    l=0.50, t=6.45, w=12.3, h=0.45, font_size=10, bold=False, color=RED_DARK)
print("Slide 3 done")


# ==============================================================================
# Slide 4 — Modèle et formules : T₂ indépendance
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "2. Model and formulas")

# Method box (kept from original)
textbox(slide, [
    "For each pair of phenotypes:",
    "  • 500 Latin Hypercube samples per phenotype  (LHS, d = 4 parameters)",
    "  • Delaunay interpolation on the LUT  →  signals S/S₀(b, TD)",
    "  • CNR computed at each grid point (b, TD):",
    "      CNR = |μ_A − μ_B| / √(σ_A² + σ_B²)",
], l=0.55, t=1.55, w=12.2, h=1.50, font_size=12)

section_header(slide, "Why is S_LUT T₂-independent?", l=0.55, t=3.12, w=12.2)

textbox(slide, [
    "In a PGSE sequence (TE = TD + δ) the measured signal is:",
    "    S_meas(b, TD) = S₀ × exp(−TE/T₂) × S_LUT(b, TD)",
    "After normalisation by S₀(TD):",
    "    S/S₀(b, TD)  =  S_LUT(b, TD)   ✓  T₂-independent",
    "T₂ cancels in the ratio — the LUT stores pure diffusion-weighted normalised signals.",
], l=0.55, t=3.56, w=12.2, h=1.25, font_size=12)

section_header(slide, "⚠  Noise IS T₂-dependent (in normalised units):",
               l=0.55, t=4.86, w=12.2, color=RED_DARK)

textbox(slide, [
    "    S₀(TD) ∝ exp(−TE/T₂)  →  weaker for longer TD",
    "    σ_noise(TD) = (1/SNR₀) × exp( (TE(TD) − TE_ref) / T₂ )     with  TE(TD) = TD + δ",
    "",
    "  Level 1 (no noise):   σ_total = σ_bio   →   CNR = |μ_A − μ_B| / √(σ_A² + σ_B²)",
    "  Level 2 (with noise): σ_total² = σ_bio² + σ_noise(TD)²",
], l=0.55, t=5.30, w=12.2, h=1.15, font_size=12)

print("Slide 4 done")


# ==============================================================================
# Slide 5 — Formules (suite) : réponse à Josh
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "2. Model and formulas (cont.) — Response to Josh's question on T₂")

section_header(slide, "Josh's two options:", l=0.55, t=1.55, w=12.2)
textbox(slide, [
    "  1)  Single T₂ for all phenotypes  —  simpler, cleanly isolates the TE penalty",
    "  2)  Per-pathology T₂  —  more biologically correct, but requires defining T₂ per tissue type",
], l=0.55, t=1.98, w=12.2, h=0.75, font_size=12)

section_header(slide, "Approach chosen:  option 1 — single T₂ = 100 ms  (literature value, glioma at 3T)",
               l=0.55, t=2.80, w=12.2, color=GREEN_DARK)
textbox(slide, [
    "  •  T₂ = 100 ms is a standard literature value for glioma at 3T (Gu 2021: HGG=127 ms, LGG=164 ms)",
    "  •  The LUT normalisation already makes the signal T₂-independent;",
    "     T₂ only enters through the noise floor σ_noise(TD)",
    "  •  Using a single reference T₂ isolates the TE penalty cleanly:",
    "     all differences between Level 1 and Level 2 are due to TD-dependent noise, not T₂ biology",
    "  •  This provides a clean baseline — approach 2 (per-pathology T₂) can be added",
    "     if the sensitivity study shows that the optimal (b*, TD*) changes significantly",
], l=0.55, t=3.22, w=12.2, h=1.75, font_size=12)

section_header(slide, "Validation: T₂ sensitivity study  (slides 15–16)", l=0.55, t=5.02, w=12.2)
textbox(slide, [
    "  T₂ swept over  {50, 75, 100, 200, 500, 1000, 3000} ms  at fixed SNR₀ = 40",
    "  → If (b*, TD*) is stable across realistic T₂ values, approach 1 is well-justified",
    "  → If results change significantly for T₂ ≈ 50–150 ms (realistic pathology range),",
    "     per-pathology T₂ (approach 2) becomes necessary",
    "  SNR₀ also swept: {15, 20, 30, 40, 60, 80, 100}  →  slide 16",
], l=0.55, t=5.44, w=12.2, h=1.28, font_size=12)

print("Slide 5 done")


# ==============================================================================
# Slide 6 — L1 : CNR heatmaps
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "3. L1: without noise correction — CNR heatmaps")

# fig2 very wide (2677×1037): scale to 12.3" wide
w2 = 12.3; h2 = w2 * 1037 / 2677   # ≈ 4.76"
add_pic(slide, "fig2_cnr_heatmaps",
        l_in=(W - w2) / 2, t_in=1.50, w_in=w2, h_in=h2)

textbox(slide,
    "TD = 80 ms is optimal for all pairs (upper boundary of the LUT grid). "
    "CNR ranges from 1.4 (Small cells / Fibrosis) to 12.5 (Cyst / Fibrosis). "
    "The LUT does not include T₂ relaxation — Level 2 corrects this.",
    l=0.55, t=1.50 + h2 + 0.10, w=12.2, h=0.55, font_size=10, color=GREY_TEXT)
print("Slide 6 done")


# ==============================================================================
# Slide 7 — L1 : signal decay
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "3. L1: without noise correction — Signal decay per phenotype")

add_pic_centered(slide, "fig1_signal_decay", top_in=1.50, max_w_in=12.3, max_h_in=5.55)
print("Slide 7 done")


# ==============================================================================
# Slide 8 — 4.A CNR discrimination : L1 vs L2 + summary table
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tA. CNR discrimination — L1 vs L2")

# fig5: very wide (5230×1184) → 12.3" wide
w5 = 12.3; h5 = w5 * 1184 / 5230   # ≈ 2.78"
add_pic(slide, "fig5_cnr_l1_vs_l2",
        l_in=(W - w5) / 2, t_in=1.50, w_in=w5, h_in=h5)

textbox(slide, "Top row: Level 1 (diffusion only). Bottom row: Level 2 (T₂=100 ms, SNR₀=40). "
        "Cyan star = optimal (b, TD). Note TD* shifts from 80 ms (L1) to shorter TD (L2) due to noise penalty.",
        l=0.55, t=1.50 + h5 + 0.05, w=12.2, h=0.45, font_size=10, color=GREY_TEXT)

# fig6: summary table (2230×593) → 11" wide
w6 = 11.0; h6 = w6 * 593 / 2230   # ≈ 2.92"
top6 = 1.50 + h5 + 0.55
add_pic(slide, "fig6_summary_l1_vs_l2",
        l_in=(W - w6) / 2, t_in=top6, w_in=w6, h_in=h6)
print("Slide 8 done")


# ==============================================================================
# Slide 9 — 4.A signal distributions (violin)
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tA. CNR discrimination — signal distributions at optimal (b, TD)")

# fig3: (2683×1183) → 12.3" wide
w3 = 12.3; h3 = w3 * 1183 / 2683   # ≈ 5.42"
add_pic(slide, "fig3_violin_optimal",
        l_in=(W - w3) / 2, t_in=1.50, w_in=w3, h_in=h3)
caption(slide,
    "Signal distributions at each pair's optimal (b, TD) from Level 1. "
    "Title of each panel shows optimal b, TD, and CNR value.",
    l=0.55, t=1.50 + h3 + 0.07, w=12.2)
print("Slide 9 done")


# ==============================================================================
# Slide 10 — 4.B Sensitivity map (raw)
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tB. Sensitivity maps — raw  |∂S/∂θ|")

add_pic_centered(slide, "fig_sens_all_raw", top_in=1.50, max_w_in=12.5, max_h_in=5.60)
caption(slide,
    "Raw sensitivity |∂S/∂θ| at each (b, TD) for each phenotype (columns) and parameter f, D_ex, r_mean (rows). "
    "Cyan star = argmax. High values → informative acquisition point.",
    l=0.55, t=7.08, w=12.2)
print("Slide 10 done")


# ==============================================================================
# Slide 11 — 4.B Sensitivity map (effective)
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tB. Sensitivity maps — effective  |∂S/∂θ| / σ_noise(TD)")

add_pic_centered(slide, "fig_sens_all_eff", top_in=1.50, max_w_in=12.5, max_h_in=5.60)
caption(slide,
    "Noise-normalised sensitivity: shows which (b, TD) are truly informative once the noise floor is accounted for. "
    "σ_noise(TD) = (1/SNR₀)·exp((TE(TD)−TE_ref)/T₂). Longer TD = higher noise = lower effective sensitivity.",
    l=0.55, t=7.08, w=12.2)
print("Slide 11 done")


# ==============================================================================
# Slide 12 — 4.C CRLB — single-b heatmap
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tC. CRLB — single-b heatmap")

add_pic_centered(slide, "fig_crlb_heatmap", top_in=1.50, max_w_in=12.5, max_h_in=5.60)
caption(slide,
    "sqrt(CRLB) = minimum std of unbiased estimator at each (b, TD). "
    "Cyan star = argmin (best precision). Low = good. "
    "T₂=100 ms, SNR₀=40. Single-b CRLB optimal ≡ effective sensitivity optimal.",
    l=0.55, t=7.08, w=12.2)
print("Slide 12 done")


# ==============================================================================
# Slide 13 — 4.C CRLB — multi-b vs TD
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tC. CRLB — relative precision vs TD  (multi-b)")

add_pic_centered(slide, "fig_crlb_vs_td", top_in=1.50, max_w_in=12.5, max_h_in=5.70)
caption(slide,
    "(a) Marginal CRLB — each parameter estimated independently.  "
    "(b) Joint CRLB via FIM⁻¹ — simultaneous estimation.  "
    "(c) Joint/marginal ratio (log scale) — correlation penalty. Vertical bars = optimal TD per parameter.",
    l=0.55, t=7.08, w=12.2)
print("Slide 13 done")


# ==============================================================================
# Slide 14 — 4.C CRLB — FIM correlation
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "4. Method with noise correction\n\tC. CRLB — FIM correlation matrix")

add_pic_centered(slide, "fig_crlb_correlation", top_in=1.50, max_w_in=12.5, max_h_in=5.60)
caption(slide,
    "FIM normalised correlation |C_ij| per phenotype × TD. "
    "Off-diagonal → 1: f, D_ex, r_mean correlated → not jointly identifiable. "
    "Off-diagonal → 0: parameters independently estimable.",
    l=0.55, t=7.08, w=12.2)
print("Slide 14 done")


# ==============================================================================
# Slide 15 — 5. T₂ sensitivity
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "5. T₂ sensitivity study")

# Portrait images side by side: fig_t2_optimum (1715×3302) + fig_t2_heatmaps (2475×3601)
H_img  = 5.50
w_opt,  _ = pic_dims("fig_t2_optimum",  max_w_in=5.0, max_h_in=H_img)
w_heat, _ = pic_dims("fig_t2_heatmaps", max_w_in=6.5, max_h_in=H_img)
gap    = 0.30
total  = w_opt + gap + w_heat
left_a = (W - total) / 2
left_b = left_a + w_opt + gap
top_img = 1.62

add_pic(slide, "fig_t2_optimum",  l_in=left_a, t_in=top_img, w_in=w_opt, h_in=H_img)
add_pic(slide, "fig_t2_heatmaps", l_in=left_b, t_in=top_img, w_in=w_heat, h_in=H_img)
caption(slide,
    "Left: optimal (b*, TD*, peak CNR) per phenotype pair for T₂ ∈ {50, 75, 100, 200, 500, 1000, 3000} ms "
    "+ Level 1 (red border, no noise).   Right: CNR heatmaps per T₂ — cyan star = optimal (b, TD).",
    l=0.55, t=top_img + H_img + 0.05, w=12.2)
print("Slide 15 done")


# ==============================================================================
# Slide 16 — 6. SNR₀ sensitivity
# ==============================================================================
slide = new_slide(LAY_CONTENT)
set_title(slide, "6. SNR₀ sensitivity study")

H_img  = 5.50
w_opt,  _ = pic_dims("fig_snr0_optimum",  max_w_in=5.0, max_h_in=H_img)
w_heat, _ = pic_dims("fig_snr0_heatmaps", max_w_in=6.5, max_h_in=H_img)
gap    = 0.30
total  = w_opt + gap + w_heat
left_a = (W - total) / 2
left_b = left_a + w_opt + gap
top_img = 1.62

add_pic(slide, "fig_snr0_optimum",  l_in=left_a, t_in=top_img, w_in=w_opt, h_in=H_img)
add_pic(slide, "fig_snr0_heatmaps", l_in=left_b, t_in=top_img, w_in=w_heat, h_in=H_img)
caption(slide,
    "Left: optimal (b*, TD*, peak CNR) per pair for SNR₀ ∈ {15, 20, 30, 40, 60, 80, 100}  "
    "+ Level 1 (red border).  T₂ = 100 ms fixed.   Right: CNR heatmaps per SNR₀.",
    l=0.55, t=top_img + H_img + 0.05, w=12.2)
print("Slide 16 done")


# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(OUT_PATH)
print(f"\nSaved -> {OUT_PATH}")
print(f"Total slides: {len(prs.slides)}")
