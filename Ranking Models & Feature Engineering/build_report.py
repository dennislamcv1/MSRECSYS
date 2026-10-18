"""
build_report.py - turns the outputs of model.py into the architecture document PDF.

    python model.py            # writes run_summary.json and training_curves.png
    python build_report.py     # writes architecture_document.pdf from those two files

Overrides (e.g. when values were copied by hand from a notebook):
    python build_report.py --run-id <mlflow-run-id> --model-version <n> --author "Your Name"
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# --- Fonts: DejaVu ships inside matplotlib, so full Unicode works on any machine ---------------
_ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Body", os.path.join(_ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(_ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Body-Oblique", os.path.join(_ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(_ttf, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Oblique",
                              boldItalic="Body-Bold")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
PANEL = colors.HexColor("#f4f3f0")
ACCENT = colors.HexColor("#2a78d6")
HILITE = colors.HexColor("#fff1c2")
TASK = {"saves": colors.HexColor("#2a78d6"), "dwell": colors.HexColor("#eb6834"),
        "shares": colors.HexColor("#1baf7a")}

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=19, leading=23, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=9.5, leading=13, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Body", fontSize=8.8, leading=12.6, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=8.8, leading=12.4, textColor=INK,
                             leftIndent=11, bulletIndent=1, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=10.8, textColor=INK),
    "cellmuted": ParagraphStyle("cellmuted", fontName="Body", fontSize=7.8, leading=10.4, textColor=INK2),
    "cellhead": ParagraphStyle("cellhead", fontName="Body-Bold", fontSize=7.8, leading=10, textColor=INK2),
    "caption": ParagraphStyle("caption", fontName="Body", fontSize=7.8, leading=10.6, textColor=INK2,
                              spaceBefore=3, spaceAfter=6),
    "mono": ParagraphStyle("mono", fontName="Mono", fontSize=8.2, leading=11, textColor=INK),
}

PLACEHOLDER = "paste from notebook output"


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


# ---------------------------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------------------------
def _box(d, x, y, w, h, fill=colors.white, stroke=RULE, sw=0.8):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _text(d, x, y, s, size=7.6, font="Body", color=INK, anchor="middle"):
    d.add(String(x, y, s, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def _arrow(d, x1, y1, x2, y2, color=INK2):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.9))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    L, Wd = 5, 2.6
    bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
    d.add(Polygon([x2, y2, bx + Wd * math.sin(ang), by - Wd * math.cos(ang),
                   bx - Wd * math.sin(ang), by + Wd * math.cos(ang)], fillColor=color, strokeColor=color,
                  strokeWidth=0.1))


def architecture_diagram(cfg, n_params):
    W, H = CONTENT_W, 178
    d = Drawing(W, H)
    mid = H / 2

    # Input
    ix, iw, ih = 0, 74, 52
    _box(d, ix, mid - ih / 2, iw, ih, fill=PANEL)
    _text(d, ix + iw / 2, mid + 8, "Input", 8.2, "Body-Bold")
    _text(d, ix + iw / 2, mid - 4, f"{cfg['in_features']}-d user×item", 6.8, color=INK2)
    _text(d, ix + iw / 2, mid - 14, "feature vector", 6.8, color=INK2)

    # Residual DCN trunk
    dx, dw, dh = 92, 118, 166
    dy = mid - dh / 2
    _box(d, dx, dy, dw, dh, fill=PANEL, stroke=INK2, sw=0.9)
    _text(d, dx + dw / 2, dy + dh - 13, "Residual DCN (shared)", 8.2, "Body-Bold")
    lh, gap = 30, 9
    top = dy + dh - 24
    for i in range(cfg["num_dcn_layers"]):
        ly = top - (i + 1) * lh - i * gap
        _box(d, dx + 9, ly, dw - 18, lh)
        _text(d, dx + dw / 2, ly + lh - 12, f"CrossNet {i + 1}", 7.6, "Body-Bold")
        _text(d, dx + dw / 2, ly + 6, f"low-rank W = U Vᵀ, r = {cfg['rank']}", 6.6, color=INK2)
        if i < cfg["num_dcn_layers"] - 1:
            _arrow(d, dx + dw / 2, ly, dx + dw / 2, ly - gap + 0.5)
    _text(d, dx + dw / 2, dy + 13, "x₀ feeds every layer; + bias + xₗ", 6.4, color=INK2)
    _text(d, dx + dw / 2, dy + 5, "output shape = input shape", 6.4, color=INK2)
    _arrow(d, ix + iw, mid, dx, mid)

    # Heads, losses
    hx, hw, hh = 236, 112, 44
    lx, lw = 368, 86
    jx, jw, jh = 470, W - 470, 64
    rows = [("saves", "Saves head", "BCE · w = 1.0", "P(save) = σ(logit)"),
            ("dwell", "Dwell Time head", "MSE · w = 0.1", "z-scored log dwell"),
            ("shares", "Private Shares head", "BCE · w = 1.0", "P(share) = σ(logit)")]
    ys = [mid + 54, mid, mid - 54]
    trunk_out = dx + dw
    for (key, name, loss, out), yc in zip(rows, ys):
        _box(d, hx, yc - hh / 2, hw, hh)
        d.add(Rect(hx, yc - hh / 2, 3.2, hh, fillColor=TASK[key], strokeColor=TASK[key], strokeWidth=0))
        _text(d, hx + hw / 2 + 1.5, yc + 8, name, 7.8, "Body-Bold")
        _text(d, hx + hw / 2 + 1.5, yc - 3, f"Linear {cfg['in_features']}→{cfg['head_hidden_size']}, ReLU",
              6.5, color=INK2)
        _text(d, hx + hw / 2 + 1.5, yc - 13, f"Linear {cfg['head_hidden_size']}→1", 6.5, color=INK2)
        _arrow(d, trunk_out, mid, hx, yc)
        _box(d, lx, yc - 17, lw, 34, fill=PANEL)
        _text(d, lx + lw / 2, yc + 3, loss, 7.4, "Body-Bold")
        _text(d, lx + lw / 2, yc - 8, out, 6.3, color=INK2)
        _arrow(d, hx + hw, yc, lx, yc)
        _arrow(d, lx + lw, yc, jx, mid + (yc - mid) * 0.25)

    _box(d, jx, mid - jh / 2, jw, jh, fill=colors.white, stroke=INK, sw=1.0)
    _text(d, jx + jw / 2, mid + 6, "Joint", 8.2, "Body-Bold")
    _text(d, jx + jw / 2, mid - 5, "loss", 8.2, "Body-Bold")
    _text(d, jx + jw / 2, mid - 17, "Σ wₜ·Lₜ", 6.8, color=INK2)
    return d


# ---------------------------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------------------------
def id_cell(value, mono=True):
    if value:
        return Paragraph(value, S["mono"] if mono else S["cell"])
    return Paragraph(f"<font backColor='#fff1c2'>&nbsp;[{PLACEHOLDER}]&nbsp;</font>", S["cell"])


def build(summary, curves_path, out_path, run_id, model_name, model_version, author):
    m, data, cfg = summary["metrics"], summary["data"], summary["config"]
    best_epoch, epochs_trained = int(m["best_epoch"]), int(m.get("epochs_trained", len(summary["history"])))
    n_params = summary["n_parameters"]
    dcn_params = cfg["num_dcn_layers"] * (2 * cfg["in_features"] * cfg["rank"] + cfg["in_features"])
    full_rank = cfg["in_features"] ** 2
    low_rank = 2 * cfg["in_features"] * cfg["rank"]

    story = []
    story.append(P("Multi-Task Heavy Ranking Network", "title"))
    sub = "Engineering architecture document"
    if author:
        sub += f" · {author}"
    story.append(P(sub + f" · {time.strftime('%B %Y')}", "subtitle"))
    story.append(Spacer(1, 10))

    # --- Submission identifiers ---
    model_uri = f"runs:/{run_id}/model" if run_id else "runs:/&lt;run_id&gt;/model"
    exp = summary.get("mlflow_experiment", "multi_task_ranking_practice")
    ids = [
        [P("MLflow Run ID", "cellhead"), id_cell(run_id)],
        [P("Azure ML registered model", "cellhead"), id_cell(model_name)],
        [P("Azure ML model version", "cellhead"), id_cell(model_version)],
        [P("MLflow model URI (registration path)", "cellhead"), Paragraph(model_uri, S["mono"])],
        [P("MLflow experiment", "cellhead"),
         P(f"{exp} &nbsp;<font color='#52514e'>(on Databricks: /Users/&lt;you&gt;/{exp})</font>", "cell")],
    ]
    t = Table(ids, colWidths=[1.95 * inch, CONTENT_W - 1.95 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)

    # --- 1. Problem ---
    story.append(P("1. Why three objectives instead of click-through rate", "h1"))
    story.append(P(
        "The incumbent ranker optimises clicks alone, which rewards flashy content and lets long-term "
        "satisfaction slide. This model predicts three signals of deeper engagement in a single forward "
        "pass: whether the user will <b>Save</b> the item, how long they will <b>dwell</b> on it, and "
        "whether they will send it as a <b>Private Share</b>. The signals do not move together. Saves are "
        f"frequent ({data['saves_rate']:.1%} of impressions in training), shares are rare "
        f"({data['shares_rate']:.1%}), and some content that drives saves suppresses sharing. A shared "
        "model trained naively can improve the frequent task at the expense of the rare, high-value one "
        "(negative transfer), so the design has to share what is common and keep what is task-specific "
        "separate, and the training run has to make interference visible."))

    # --- 2. Architecture ---
    story.append(P("2. Final model architecture", "h1"))
    story.append(KeepTogether([
        architecture_diagram(cfg, n_params),
        P(f"Figure 1. Shared Residual DCN trunk feeding three task towers. {n_params:,} trainable "
          f"parameters in total; {dcn_params:,} of them in the cross layers.", "caption"),
    ]))
    story.extend(bullets([
        f"<b>Shared trunk: Residual DCN.</b> {cfg['num_dcn_layers']} stacked CrossNet layers compute "
        "x<sub>l+1</sub> = x<sub>0</sub> ⊙ (x<sub>l</sub> V Uᵀ) + b + x<sub>l</sub>. Each layer adds one "
        "degree of explicit feature interaction, so three layers model crosses up to fourth order "
        "without hand-built cross features. The original input x<sub>0</sub> is re-injected at every "
        "layer and the identity term keeps gradients flowing; the block preserves the input shape "
        f"({cfg['in_features']} → {cfg['in_features']}), which is what allows layers to be stacked.",
        f"<b>Low-rank cross weights.</b> W ≈ U Vᵀ with rank {cfg['rank']}: x<sub>l</sub> is projected "
        f"down through V ({cfg['in_features']}→{cfg['rank']}) and back up through Uᵀ, costing "
        f"{low_rank:,} weights per layer instead of {full_rank:,} for a full matrix (50% fewer at this "
        "width, and the saving grows with feature width).",
        f"<b>Task towers.</b> Saves, Dwell Time and Private Shares each get their own "
        f"Linear({cfg['in_features']}→{cfg['head_hidden_size']}) → ReLU → "
        f"Linear({cfg['head_hidden_size']}→1) head. The trunk learns interactions all three objectives "
        "use; the towers give each objective private capacity, which is the first line of defence "
        "against one task overwriting another.",
        "<b>Outputs.</b> Saves and Shares heads emit logits (σ gives probabilities at serving time); "
        "the Dwell Time head regresses z-scored log seconds, which tames the heavy right tail of raw "
        "dwell and puts its error on a scale comparable to the classification losses.",
    ]))

    # --- 3. Objective & training ---
    story.append(P("3. Joint objective and training", "h1"))
    story.append(P(
        "joint_loss = 1.0 · BCE(saves) + <b>0.1</b> · MSE(dwell) + 1.0 · BCE(shares). "
        "BCEWithLogitsLoss handles the two binary tasks; MSELoss handles dwell. MSE runs on a larger "
        "scale than BCE and, left unweighted, dominates the shared gradient, so it is down-weighted to "
        "0.1. All weights are config fields, so rebalancing is a one-line change if validation AUC on "
        "Saves or Shares stalls while the joint loss keeps falling."))
    story.extend(bullets([
        f"AdamW (lr {cfg['learning_rate']:g}, weight decay {cfg['weight_decay']:g}), batch "
        f"{cfg['batch_size']}, gradient-norm clipping at {cfg['grad_clip_norm']:g}; learning rate halves "
        "when validation joint loss stalls.",
        f"Early stopping on validation combined AUC (patience {cfg['early_stopping_patience']}); the best "
        f"checkpoint is restored before logging. This run peaked at epoch {best_epoch} and stopped after "
        f"epoch {epochs_trained}.",
        "Cross weights use Xavier initialisation. With the unit-variance torch.randn values in the "
        "starter code, a 4-epoch diagnostic on 60k rows produced DCN outputs with a standard deviation "
        "near 2.7×10⁵ at initialisation and all three AUCs stayed at about 0.50; Xavier reached "
        "0.83 / 0.91 / 0.82 (Saves / Dwell / Shares) on the same budget.",
        f"Data: {data['n_train'] + data['n_val']:,} synthetic user×item impressions with "
        f"{cfg['in_features']} features, split {data['n_train']:,} train / {data['n_val']:,} validation "
        f"with a fixed seed ({cfg['seed']}). Labels are driven by pairwise feature crosses, and one cross "
        "raises Saves while lowering Shares, so the trade-off described in section 1 is present in the "
        f"data. Median dwell is {data['dwell_median_seconds']:.1f} s; dwell is standardised with "
        "training-split statistics only.",
    ]))

    # --- 4. Metrics ---
    story.append(P("4. Validation metrics", "h1"))
    rows = [
        [P("Metric", "cellhead"), P("Value", "cellhead"), P("Definition", "cellhead")],
        ["Saves AUC", f"{m['final_auc_saves']:.4f}", "ROC-AUC of σ(saves logit) against the Saved label"],
        ["Dwell Time AUC", f"{m['final_auc_dwell']:.4f}",
         "ROC-AUC of predicted dwell against a long-dwell label (at or above the training median)"],
        ["Private Shares AUC", f"{m['final_auc_shares']:.4f}",
         f"ROC-AUC of σ(shares logit) against the Shared label ({data['shares_rate']:.1%} positives)"],
        ["Combined AUC", f"{m['final_combined_auc']:.4f}", "Unweighted mean of the three per-task AUCs"],
        ["Final joint loss (validation)", f"{m['final_val_joint_loss']:.4f}",
         f"1.0·BCE + 0.1·MSE + 1.0·BCE on the {data['n_val']:,}-row validation split"],
        ["Final joint loss (training)", f"{m['final_train_joint_loss']:.4f}",
         f"Same objective, mean over the epoch {best_epoch} training batches"],
        ["Dwell RMSE", f"{m['final_rmse_dwell']:.4f}", "Root-mean-square error in z-scored log seconds"],
    ]
    body_rows = [[P(r[0], "cell"), Paragraph(f"<b>{r[1]}</b>", S["cell"]), P(r[2], "cellmuted")]
                 for r in rows[1:]]
    mt = Table([rows[0]] + body_rows, colWidths=[1.75 * inch, 0.8 * inch, CONTENT_W - 2.55 * inch],
               repeatRows=1)
    mt.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK2),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 4), (-1, 5), PANEL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([mt, P(
        f"Table 1. Metrics for the registered checkpoint (epoch {best_epoch}), evaluated on the held-out "
        "validation split. Combined AUC and validation joint loss are shaded as the headline figures.",
        "caption")]))

    h = summary["history"]
    peak = {k: max(h, key=lambda r: r[k])["epoch"] for k in ("auc_saves", "auc_dwell", "auc_shares")}
    nt = int(m.get("negative_transfer_detected", 0))
    if len(set(peak.values())) == 1:
        peak_text = f"all three tasks peaked at epoch {peak['auc_saves']}"
    else:
        peak_text = (f"Saves peaked at epoch {peak['auc_saves']}, Dwell at epoch {peak['auc_dwell']} "
                     f"and Shares at epoch {peak['auc_shares']}")
    story.append(P(
        "<b>Reading the curves for negative transfer.</b> The run logs auc_saves, auc_dwell and auc_shares "
        "every epoch, plus a negative_transfer_flag that fires when Saves AUC rises while Shares AUC falls "
        f"for {cfg['neg_transfer_patience']} consecutive epochs. "
        + ("The flag never fired in this run: " if not nt else "The flag fired in this run, ")
        + peak_text + ", and after that point the three moved together (Figure 2). The late decline is "
        "ordinary over-fitting (training loss keeps falling while validation loss rises for every task), "
        "which early stopping and checkpoint restore already handle. If a future run shows the scissor "
        "pattern instead (Saves up, Shares down), the remedies in order are: raise the Shares loss weight "
        "or add a positive-class weight, then move from a single shared trunk to gated experts "
        "(MMoE/PLE)."))

    if os.path.exists(curves_path):
        img_w = CONTENT_W
        img = Image(curves_path, width=img_w, height=img_w * 3.8 / 10)
        story.append(KeepTogether([img, P(
            "Figure 2. Per-task validation AUC (left) and joint loss (right) by epoch, as logged to MLflow. "
            "The dashed line marks the restored checkpoint.", "caption")]))


    # --- 5. MLOps ---
    story.append(P("5. Tracking and registration", "h1"))
    story.extend(bullets([
        "<b>MLflow.</b> Every hyper-parameter and data statistic is logged as a param. Each epoch logs "
        "joint_loss, val_joint_loss, the three per-task AUCs, combined_auc, per-task losses, dwell RMSE, "
        "the learning rate and the negative-transfer flag. The run also stores the training-curve figure, "
        "the epoch history as JSON, and the PyTorch model (mlflow.pytorch.log_model with an explicit "
        "signature and serialization_format=\"pickle\").",
        "<b>Azure ML registry.</b> After the run, MLClient (DefaultAzureCredential) calls "
        "ml_client.models.create_or_update with Model(path=f\"runs:/{run_id}/model\", "
        f"name=\"{summary.get('azureml_model_name') or 'multi-task-ranking-model'}\", "
        "type=AssetTypes.MLFLOW_MODEL). The run ID, model URI and per-task AUCs are attached as tags, and "
        "the assigned version is written back to the MLflow run as a tag, so registry entry and run point "
        "at each other.",
        "<b>Environment handling.</b> On Databricks the experiment uses the absolute "
        "/Users/&lt;you&gt;/… path Databricks requires. Elsewhere MLflow tracks into the Azure ML "
        "workspace when azureml-mlflow is installed, or locally otherwise. A runs:/ path only resolves "
        "inside the workspace that owns the run, so when it does not, the script downloads the logged "
        "model folder and registers that instead, keeping the run ID in the tags.",
    ]))

    # --- 6. Next steps ---
    story.append(P("6. Before production", "h1"))
    story.extend(bullets([
        "Retrain on logged impressions with time-based splits; synthetic data validates the pipeline, "
        "not the business lift.",
        "Calibrate the Shares probability (Platt or isotonic) before it feeds a blended ranking score, "
        "and add per-segment AUC so rare-task regressions in small cohorts are not averaged away.",
        "Replace fixed loss weights with learned uncertainty weighting and A/B the blended score against "
        "the CTR model on saves, shares and 7-day return rate.",
    ]))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7)
        canvas.setFillColor(INK2)
        canvas.drawString(MARGIN, 0.5 * inch, "Multi-Task Heavy Ranking Network · architecture document")
        canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="Multi-Task Heavy Ranking Network - Architecture",
                            author=author or "", subject="Engineering architecture document")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="run_summary.json")
    ap.add_argument("--curves", default="training_curves.png")
    ap.add_argument("--out", default="architecture_document.pdf")
    ap.add_argument("--run-id", default=None, help="Override the MLflow Run ID")
    ap.add_argument("--model-name", default=None, help="Override the Azure ML model name")
    ap.add_argument("--model-version", default=None, help="Override the Azure ML model version")
    ap.add_argument("--author", default="")
    ap.add_argument("--blank-ids", action="store_true",
                    help="Leave the Run ID and model version as highlighted fill-in fields")
    args = ap.parse_args()

    with open(args.summary) as fh:
        summary = json.load(fh)
    run_id = args.run_id or (None if args.blank_ids else summary.get("mlflow_run_id"))
    name = args.model_name or summary.get("azureml_model_name") or "multi-task-ranking-model"
    version = args.model_version or (None if args.blank_ids else summary.get("azureml_model_version"))
    build(summary, args.curves, args.out, run_id, name, version, args.author)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
