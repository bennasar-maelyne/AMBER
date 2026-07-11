"""
Script de mise à jour de la présentation DWI acquisition optimisation.pptx
Répond au mail de Josh sur T2 et complète toutes les slides vides avec les figures.
"""

import os
import copy
from lxml import etree

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))

PPTX_IN  = os.path.join(THIS_DIR, "Figures", "DWI acquisition optimisation.pptx")
PPTX_OUT = os.path.join(THIS_DIR, "Figures", "DWI acquisition optimisation.pptx")

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

prs = Presentation(PPTX_IN)
W = prs.slide_width.inches    # 13.33"
H = prs.slide_height.inches   # 7.50"
LAYOUT_TITLE_CONTENT = prs.slide_layouts[1]   # "Titre et contenu"


# ── Helpers ────────────────────────────────────────────────────────────────────

def fit_image(img_w_px, img_h_px, max_w_in, max_h_in):
    """Return (width_in, height_in) scaled to fit in box while preserving aspect."""
    ratio = img_w_px / img_h_px
    w = min(max_w_in, max_h_in * ratio)
    h = w / ratio
    if h > max_h_in:
        h = max_h_in
        w = h * ratio
    return w, h


def center_x(w_in):
    return (W - w_in) / 2


def add_picture_centered(slide, img_path, top_in, max_w_in, max_h_in):
    """Add an image centered horizontally, scaled to fit the given box."""
    from PIL import Image
    img = Image.open(img_path)
    iw, ih = img.size
    w, h = fit_image(iw, ih, max_w_in, max_h_in)
    left = center_x(w)
    slide.shapes.add_picture(img_path, Inches(left), Inches(top_in), Inches(w), Inches(h))
    return h


def add_textbox(slide, text_lines, left, top, width, height,
                font_size=13, bold=False, color=None, align=PP_ALIGN.LEFT,
                line_spacing=1.15):
    from pptx.oxml import parse_xml
    from pptx.oxml.ns import nsmap
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for pi, line in enumerate(text_lines):
        p = tf.paragraphs[0] if pi == 0 else tf.add_paragraph()
        p.alignment = align
        # line spacing
        p.space_after = Pt(2)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        if color:
            run.font.color.rgb = RGBColor(*color)
    return txBox


def clear_placeholder_text(slide, ph_idx=1):
    """Remove all text from a content placeholder."""
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == ph_idx:
            tf = shape.text_frame
            for para in tf.paragraphs:
                for run in para.runs:
                    run.text = ""
            # clear all paragraphs except first
            for para in list(tf.paragraphs)[1:]:
                p_elem = para._p
                p_elem.getparent().remove(p_elem)
            tf.paragraphs[0].text = ""
            return shape
    return None


def set_slide_title(slide, title_text):
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            shape.text = title_text
            return


def insert_slide_after(prs, after_idx, layout):
    """Insert a new slide (using layout) right after after_idx (0-based)."""
    new_slide = prs.slides.add_slide(layout)
    # The new slide is at the end of the XML id list; move it to after_idx+1
    xml_slides = prs.slides._sldIdLst
    ids = list(xml_slides)
    new_id = ids[-1]
    xml_slides.remove(new_id)
    xml_slides.insert(after_idx + 1, new_id)
    return new_slide


def get_slide_idx(prs, slide):
    """Return the current 0-based index of a slide object."""
    for i, s in enumerate(prs.slides):
        if s == slide:
            return i
    raise ValueError("Slide not found")


# ── SLIDE 4: "2. Model and formulas" ──────────────────────────────────────────
# Répondre au mail de Josh : T2 indépendance dans la LUT, bruit dépendant de T2

slide4 = prs.slides[3]
clear_placeholder_text(slide4, ph_idx=1)

# Box 1 — Titre de section (gras, couleur accent)
add_textbox(slide4,
    ["Signal model: why S_LUT is T2-independent"],
    left=0.55, top=1.55, width=12.2, height=0.45,
    font_size=14, bold=True, color=(31, 73, 125))

# Box 2 — Formules signal
add_textbox(slide4,
    [
        "The LUT stores normalised signals S/S₀(b, TD) — purely diffusion-weighted:",
        "",
        "    S_measured(b, TD)  =  S₀ × exp(−TE/T₂)  ×  S_LUT(b, TD)",
        "    S_normalised(b, TD)  =  S_measured / S₀  =  S_LUT(b, TD)       ✓ T₂-independent",
        "",
        "T₂ cancels in the ratio → the LUT signal is independent of T₂.",
    ],
    left=0.55, top=2.05, width=12.2, height=1.55,
    font_size=12)

# Box 3 — Bruit dépendant de T2
add_textbox(slide4,
    ["⚠  Thermal noise is NOT T₂-independent (in normalised units):"],
    left=0.55, top=3.65, width=12.2, height=0.35,
    font_size=14, bold=True, color=(192, 0, 0))

add_textbox(slide4,
    [
        "    S₀(TD) ∝ exp(−TE/T₂)  decreases for longer TD  (TE = TD + δ)",
        "    σ_noise(TD) = σ_abs / S₀(TD) = (1/SNR₀) × exp( (TE(TD) − TE_ref) / T₂ )",
        "",
        "    → longer TD = weaker S₀ = larger relative noise floor",
    ],
    left=0.55, top=4.05, width=12.2, height=1.10,
    font_size=12)

# Box 4 — Level 1 vs Level 2
add_textbox(slide4,
    ["Two analysis levels:"],
    left=0.55, top=5.20, width=12.2, height=0.35,
    font_size=14, bold=True, color=(31, 73, 125))

add_textbox(slide4,
    [
        "  Level 1 — diffusion only (no noise):   CNR = |μ_A − μ_B| / √(σ_A² + σ_B²)",
        "  Level 2 — T₂ + thermal noise:          σ_total(TD)² = σ_bio² + σ_noise(TD)²",
        "                                         CNR = |μ_A − μ_B| / √(σ_A,total² + σ_B,total²)",
    ],
    left=0.55, top=5.58, width=12.2, height=0.88,
    font_size=12)

print("Slide 4 updated")


# ── SLIDE 5: Extra formulas — réponse directe au mail de Josh ─────────────────

slide5 = prs.slides[4]
set_slide_title(slide5, "2. Model and formulas (cont.)")
clear_placeholder_text(slide5, ph_idx=1)

add_textbox(slide5,
    ["Response to Josh's question: which T₂ approach?"],
    left=0.55, top=1.55, width=12.2, height=0.40,
    font_size=14, bold=True, color=(31, 73, 125))

add_textbox(slide5,
    [
        "Josh proposed two options:",
        "  1) Single T₂ for all phenotypes  —  simpler, isolates the TE penalty effect",
        "  2) Per-pathology T₂  —  more biologically correct, but requires T₂ per tissue type",
    ],
    left=0.55, top=2.00, width=12.2, height=0.90,
    font_size=12)

add_textbox(slide5,
    ["Approach chosen: option 1 — single T₂ = 100 ms"],
    left=0.55, top=2.95, width=12.2, height=0.38,
    font_size=13, bold=True, color=(0, 112, 0))

add_textbox(slide5,
    [
        "  •  T₂ = 100 ms is a standard literature value for glioma at 3T",
        "  •  The LUT normalisation already makes the signal T₂-independent;",
        "     T₂ only enters through the noise floor σ_noise(TD)",
        "  •  Using a single reference T₂ isolates the TE penalty cleanly:",
        "     all differences between Level 1 and Level 2 are due to TD-dependent noise",
    ],
    left=0.55, top=3.38, width=12.2, height=1.35,
    font_size=12)

add_textbox(slide5,
    ["Validation: T₂ sensitivity study"],
    left=0.55, top=4.78, width=12.2, height=0.38,
    font_size=13, bold=True, color=(31, 73, 125))

add_textbox(slide5,
    [
        "  T₂ swept over {50, 75, 100, 200, 500, 1000, 3000} ms at fixed SNR₀ = 40",
        "  → If optimal (b*, TD*) is stable across T₂, approach 1 is well-justified",
        "  → If results differ for T₂ ≈ realistic pathology values,",
        "     per-pathology T₂ (approach 2) would become necessary",
    ],
    left=0.55, top=5.20, width=12.2, height=1.15,
    font_size=12)

print("Slide 5 updated")


# ── SLIDE 8: "4.A CNR discrimination" — fig5 + fig6 ──────────────────────────

slide8 = prs.slides[7]
clear_placeholder_text(slide8, ph_idx=1)

# fig5: very wide (5230×1184) — scale to 12" wide
w5 = 12.0; h5 = w5 * 1184 / 5230   # ≈ 2.72"
slide8.shapes.add_picture(FIGS["fig5_cnr_l1_vs_l2"],
                           Inches(center_x(w5)), Inches(1.60),
                           Inches(w5), Inches(h5))

# fig6: summary table (2230×593) — scale to 10" wide
w6 = 10.0; h6 = w6 * 593 / 2230     # ≈ 2.66"
slide8.shapes.add_picture(FIGS["fig6_summary_l1_vs_l2"],
                           Inches(center_x(w6)), Inches(1.60 + h5 + 0.25),
                           Inches(w6), Inches(h6))

print("Slide 8 updated")


# ── NEW SLIDE after 8: violin plots ───────────────────────────────────────────

s_idx = get_slide_idx(prs, slide8)
slide_violin = insert_slide_after(prs, s_idx, LAYOUT_TITLE_CONTENT)
set_slide_title(slide_violin, "4. Method with noise correction\n\tA. CNR discrimination — signal distributions")
clear_placeholder_text(slide_violin, ph_idx=1)

# fig3: violin (2683×1183) — wide, scale to 12"
w3 = 12.0; h3 = w3 * 1183 / 2683   # ≈ 5.29"
slide_violin.shapes.add_picture(FIGS["fig3_violin_optimal"],
                                 Inches(center_x(w3)), Inches(1.55),
                                 Inches(w3), Inches(h3))

add_textbox(slide_violin,
    ["Signal distributions at each pair's optimal (b, TD) — CNR value annotated in title of each panel"],
    left=0.55, top=1.55 + h3 + 0.08, width=12.2, height=0.45,
    font_size=10, bold=False, color=(80, 80, 80))

print("Violin slide inserted")


# ── SLIDE 9: "4.B Sensitivity map" — fig_sens_all_raw ────────────────────────

slide9 = prs.slides[get_slide_idx(prs, slide_violin) + 1]
clear_placeholder_text(slide9, ph_idx=1)

add_picture_centered(slide9, FIGS["fig_sens_all_raw"],
                     top_in=1.50, max_w_in=12.5, max_h_in=5.60)

add_textbox(slide9,
    ["Raw sensitivity |dS/dθ| at each (b, TD) grid point for each phenotype and parameter (f, D_ex, r_mean). Cyan star = argmax."],
    left=0.55, top=7.05, width=12.2, height=0.38,
    font_size=9, bold=False, color=(80, 80, 80))

print("Slide 9 updated")


# ── NEW SLIDE after 9: effective sensitivity ──────────────────────────────────

s9_idx = get_slide_idx(prs, slide9)
slide_eff = insert_slide_after(prs, s9_idx, LAYOUT_TITLE_CONTENT)
set_slide_title(slide_eff, "4. Method with noise correction\n\tB. Sensitivity map — noise-normalised")
clear_placeholder_text(slide_eff, ph_idx=1)

add_picture_centered(slide_eff, FIGS["fig_sens_all_eff"],
                     top_in=1.50, max_w_in=12.5, max_h_in=5.60)

add_textbox(slide_eff,
    ["Effective sensitivity |dS/dθ| / σ_noise(TD) — highlights which (b, TD) are truly informative after accounting for the noise floor."],
    left=0.55, top=7.05, width=12.2, height=0.38,
    font_size=9, bold=False, color=(80, 80, 80))

print("Effective sensitivity slide inserted")


# ── SLIDE 10: "4.C CRLB study" — fig_crlb_heatmap ───────────────────────────

slide10 = prs.slides[get_slide_idx(prs, slide_eff) + 1]
clear_placeholder_text(slide10, ph_idx=1)

add_picture_centered(slide10, FIGS["fig_crlb_heatmap"],
                     top_in=1.50, max_w_in=12.5, max_h_in=5.60)

add_textbox(slide10,
    ["Single-b CRLB — minimum std of estimator sqrt(CRLB) for each (b, TD). Cyan star = argmin per panel. Low = good precision."],
    left=0.55, top=7.05, width=12.2, height=0.38,
    font_size=9, bold=False, color=(80, 80, 80))

print("Slide 10 updated")


# ── NEW SLIDE after 10: CRLB vs TD ───────────────────────────────────────────

s10_idx = get_slide_idx(prs, slide10)
slide_crlb_td = insert_slide_after(prs, s10_idx, LAYOUT_TITLE_CONTENT)
set_slide_title(slide_crlb_td, "4. Method with noise correction\n\tC. CRLB study — relative precision vs TD")
clear_placeholder_text(slide_crlb_td, ph_idx=1)

add_picture_centered(slide_crlb_td, FIGS["fig_crlb_vs_td"],
                     top_in=1.50, max_w_in=12.5, max_h_in=5.70)

add_textbox(slide_crlb_td,
    ["(a) Marginal CRLB — each parameter estimated independently. (b) Joint CRLB via FIM⁻¹. (c) Joint/marginal ratio (log) — cost of correlations. Vertical bars = optimal TD per param."],
    left=0.55, top=7.05, width=12.2, height=0.38,
    font_size=9, bold=False, color=(80, 80, 80))

print("CRLB vs TD slide inserted")


# ── NEW SLIDE after CRLB TD: FIM correlation ─────────────────────────────────

s_crlb_td_idx = get_slide_idx(prs, slide_crlb_td)
slide_crlb_corr = insert_slide_after(prs, s_crlb_td_idx, LAYOUT_TITLE_CONTENT)
set_slide_title(slide_crlb_corr, "4. Method with noise correction\n\tC. CRLB study — FIM correlation")
clear_placeholder_text(slide_crlb_corr, ph_idx=1)

add_picture_centered(slide_crlb_corr, FIGS["fig_crlb_correlation"],
                     top_in=1.50, max_w_in=12.5, max_h_in=5.60)

add_textbox(slide_crlb_corr,
    ["FIM normalised correlation matrix |C_ij| per phenotype × TD. Off-diagonal ≈ 1 → parameters are correlated and not jointly identifiable at that TD."],
    left=0.55, top=7.05, width=12.2, height=0.38,
    font_size=9, bold=False, color=(80, 80, 80))

print("FIM correlation slide inserted")


# ── SLIDE "5. T2 sensitivity" ─────────────────────────────────────────────────
# Original slide 11 (idx 10). After 4 insertions (violin, eff, crlb_td, crlb_corr): idx 14.

slide_t2 = prs.slides[14]

if slide_t2:
    clear_placeholder_text(slide_t2, ph_idx=1)

    # Side-by-side: fig_t2_optimum (portrait) + fig_t2_heatmaps (portrait)
    H_img = 5.50
    w_opt  = H_img * 1715 / 3302   # ≈ 2.86"
    w_heat = H_img * 2475 / 3601   # ≈ 3.78"
    gap = 0.40
    total_w = w_opt + gap + w_heat
    left_opt  = (W - total_w) / 2
    left_heat = left_opt + w_opt + gap
    top_img   = 1.65

    slide_t2.shapes.add_picture(FIGS["fig_t2_optimum"],
                                 Inches(left_opt), Inches(top_img),
                                 Inches(w_opt), Inches(H_img))
    slide_t2.shapes.add_picture(FIGS["fig_t2_heatmaps"],
                                 Inches(left_heat), Inches(top_img),
                                 Inches(w_heat), Inches(H_img))

    add_textbox(slide_t2,
        ["Left: optimal (b*, TD*, peak CNR) per pair for T₂ ∈ {50…75…100…1000…3000} ms + L1 (red border). "
         "Right: CNR heatmaps per T₂ value — cyan star = optimal (b, TD)."],
        left=0.55, top=top_img + H_img + 0.05, width=12.2, height=0.42,
        font_size=9, bold=False, color=(80, 80, 80))

    print("T2 sensitivity slide updated")
else:
    print("WARNING: T2 sensitivity slide not found")


# ── SLIDE "6. SNR sensitivity" ────────────────────────────────────────────────
# Original slide 12 (idx 11). After 4 insertions: idx 15.

slide_snr = prs.slides[15]

if slide_snr:
    clear_placeholder_text(slide_snr, ph_idx=1)

    H_img = 5.50
    w_opt  = H_img * 1715 / 3302
    w_heat = H_img * 2486 / 3601
    gap = 0.40
    total_w = w_opt + gap + w_heat
    left_opt  = (W - total_w) / 2
    left_heat = left_opt + w_opt + gap
    top_img   = 1.65

    slide_snr.shapes.add_picture(FIGS["fig_snr0_optimum"],
                                  Inches(left_opt), Inches(top_img),
                                  Inches(w_opt), Inches(H_img))
    slide_snr.shapes.add_picture(FIGS["fig_snr0_heatmaps"],
                                  Inches(left_heat), Inches(top_img),
                                  Inches(w_heat), Inches(H_img))

    add_textbox(slide_snr,
        ["Left: optimal (b*, TD*, peak CNR) per pair for SNR₀ ∈ {15, 20, 30, 40, 60, 80, 100} + L1 (red border). "
         "T₂ = 100 ms fixed. Right: CNR heatmaps per SNR₀ — cyan star = optimal (b, TD)."],
        left=0.55, top=top_img + H_img + 0.05, width=12.2, height=0.42,
        font_size=9, bold=False, color=(80, 80, 80))

    print("SNR sensitivity slide updated")
else:
    print("WARNING: SNR sensitivity slide not found")


# ── Update table of contents (slide 2) ────────────────────────────────────────

slide2 = prs.slides[1]
for shape in slide2.placeholders:
    if shape.placeholder_format.idx == 1:
        tf = shape.text_frame
        # Clear and rewrite
        for para in tf.paragraphs:
            for run in para.runs:
                run.text = ""
        for para in list(tf.paragraphs)[1:]:
            para._p.getparent().remove(para._p)
        tf.paragraphs[0].text = ""

        lines = [
            "1. Phenotypes",
            "2. Model and formulas (T₂ independence, noise model, Level 1 & Level 2)",
            "3. L1: CNR without noise correction",
            "4. Method with noise correction:",
            "       A. CNR discrimination (L1 vs L2) + signal distributions",
            "       B. Sensitivity maps (raw and noise-normalised)",
            "       C. CRLB study (single-b, multi-b, FIM correlation)",
            "5. T₂ sensitivity study",
            "6. SNR₀ sensitivity study",
        ]
        for li, line in enumerate(lines):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.text = line
        break

print("Table of contents updated")


# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(PPTX_OUT)
print(f"\nSaved -> {PPTX_OUT}")
print(f"Total slides: {len(prs.slides)}")
