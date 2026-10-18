"""
app/reporting.py

Self-contained HTML report for one survey (no server, no internet needed):
KPIs, field map with the spray route, per-plant decisions with reasons,
Grad-CAM thumbnails and — for the synthetic demo field — a system-level
evaluation against the known ground truth.

    write_survey_report(survey_id, pipeline_result) -> "data/outputs/<id>/report.html"
"""

from __future__ import annotations

import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = REPO_ROOT / "data" / "images"
OUTPUTS_ROOT = REPO_ROOT / "data" / "outputs"

LABEL_COLORS = {
    "healthy": "#2f9e44",
    "early_blight": "#e8590c",
    "late_blight": "#c2255c",
    "septoria_leaf_spot": "#1971c2",
    "unclassified": "#868e96",
}
ACTION_TEXT = {
    "spray": "Spray", "no_spray": "No spray", "monitor": "Monitor",
    "inspect": "Inspect", "hold": "Hold (re-treatment interval)",
}


# ── Ground-truth evaluation (synthetic demo field only) ───────────────────────

def evaluate_against_ground_truth(survey_id: str) -> dict | None:
    """
    Compare pipeline output with data/images/<survey_id>/ground_truth.json
    (written by scripts/make_demo_field.py). Returns None for real surveys.
    """
    gt_path = IMAGES_ROOT / survey_id / "ground_truth.json"
    if not gt_path.exists():
        return None
    from app.db import get_connection
    from app.decision.database import get_decisions

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    conn = get_connection()
    try:
        dets = conn.execute(
            "SELECT d.plant_id, d.image_filename, (d.bbox_x1+d.bbox_x2)/2, "
            "(d.bbox_y1+d.bbox_y2)/2, c.label FROM detections d "
            "LEFT JOIN classifications c ON c.survey_id=d.survey_id AND c.plant_id=d.plant_id "
            "WHERE d.survey_id=?", (survey_id,)).fetchall()
        decisions = {d["plant_id"]: d for d in get_decisions(conn, survey_id)}
    finally:
        conn.close()

    plants = gt["plants"]
    used = set()
    matches = []
    for pid, img, cx, cy, label in dets:
        best, best_d = None, 1e9
        for j, p in enumerate(plants):
            if j in used or p.get("image_filename", img) != img:
                continue
            d = math.hypot(cx - p["pixel_x"], cy - p["pixel_y"])
            if d < best_d:
                best, best_d = j, d
        if best is not None and best_d <= gt.get("match_radius_px", 60):
            used.add(best)
            matches.append((pid, label, plants[best]))
    tp = len(matches)
    n_gt, n_det = len(plants), len(dets)
    correct = sum(1 for _, label, p in matches if label == p["true_label"])
    loc_err = []
    for pid, _, p in matches:
        dec = decisions.get(pid)
        if dec and dec.get("is_mapped"):
            loc_err.append(math.hypot(dec["field_x_m"] - p["field_x_m"],
                                      dec["field_y_m"] - p["field_y_m"]))
    diseased = [(pid, p) for pid, _, p in matches if p["true_label"] != "healthy"]
    diseased_sprayed = sum(1 for pid, _ in diseased
                           if decisions.get(pid, {}).get("action") in ("spray", "inspect",
                                                                        "monitor", "hold"))
    healthy = [(pid, p) for pid, _, p in matches if p["true_label"] == "healthy"]
    healthy_sprayed = sum(1 for pid, _ in healthy
                          if decisions.get(pid, {}).get("action") == "spray")
    return {
        "ground_truth_plants": n_gt,
        "detections": n_det,
        "matched": tp,
        "detection_recall": round(tp / n_gt, 4) if n_gt else None,
        "detection_precision": round(tp / n_det, 4) if n_det else None,
        "classification_accuracy_on_matched": round(correct / tp, 4) if tp else None,
        "classification_correct": correct,
        "mean_localisation_error_m": round(sum(loc_err) / len(loc_err), 4) if loc_err else None,
        "max_localisation_error_m": round(max(loc_err), 4) if loc_err else None,
        "diseased_plants_actioned": f"{diseased_sprayed}/{len(diseased)}",
        "healthy_plants_sprayed": f"{healthy_sprayed}/{len(healthy)}",
        "note": gt.get("note", ""),
    }


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _img_data_uri(path: Path, max_side: int = 96) -> str | None:
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None
    except Exception:
        return None


def _field_svg(decisions: list[dict], plan: dict | None, field_w: float, field_h: float) -> str:
    W = 640
    scale = W / max(field_w, 1e-6)
    H = int(field_h * scale)
    pad = 28

    def px(x, y):  # field y grows "north" -> draw with origin bottom-left
        return pad + x * scale, pad + (field_h - y) * scale

    parts = [f'<svg viewBox="0 0 {W + 2 * pad} {H + 2 * pad}" role="img" '
             f'aria-label="Field map with plants and spray route" class="fieldmap">',
             f'<rect x="{pad}" y="{pad}" width="{W}" height="{H}" class="soil"/>']
    for gx in range(int(field_w) + 1):
        x0, _ = px(gx, 0)
        parts.append(f'<line x1="{x0:.1f}" y1="{pad}" x2="{x0:.1f}" y2="{pad + H}" class="grid"/>')
        parts.append(f'<text x="{x0:.1f}" y="{pad + H + 16}" class="axis">{gx}</text>')
    for gy in range(int(field_h) + 1):
        _, y0 = px(0, gy)
        parts.append(f'<line x1="{pad}" y1="{y0:.1f}" x2="{pad + W}" y2="{y0:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad - 8}" y="{y0 + 4:.1f}" class="axis" text-anchor="end">{gy}</text>')
    if plan:
        dx, dy = plan["config"]["depot_x_m"], plan["config"]["depot_y_m"]
        for prog in plan.get("programmes", []):
            color = "#c2255c" if "late" in prog["treatment"] else "#e8590c"
            for s in prog["sorties"]:
                pts = " ".join(f"{px(x, y)[0]:.1f},{px(x, y)[1]:.1f}" for x, y in s["path"])
                parts.append(f'<polyline points="{pts}" class="route" stroke="{color}"/>')
        ddx, ddy = px(dx, dy)
        parts.append(f'<rect x="{ddx - 7:.1f}" y="{ddy - 7:.1f}" width="14" height="14" class="depot"/>')
    for d in decisions:
        if not d.get("is_mapped"):
            continue
        cx, cy = px(d["field_x_m"], d["field_y_m"])
        color = LABEL_COLORS.get(d["disease_label"], "#868e96")
        title = html.escape(f"Plant {d['plant_id']}: {d['disease_label']} — "
                            f"{ACTION_TEXT.get(d['action'], d['action'])}")
        if d["action"] == "spray":
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" class="spray-ring" '
                         f'stroke="{color}"/>')
        r = 7 if d["action"] != "inspect" else 6
        shape = (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}">'
                 if d["action"] != "inspect" else
                 f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" '
                 f'fill="{color}" transform="rotate(45 {cx:.1f} {cy:.1f})">')
        tag = "circle" if d["action"] != "inspect" else "rect"
        parts.append(f'{shape}<title>{title}</title></{tag}>')
        parts.append(f'<text x="{cx:.1f}" y="{cy - 12:.1f}" class="pid">{d["plant_id"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-sub">{html.escape(sub)}</div></div>')


def write_survey_report(survey_id: str, pipeline_result: dict | None = None) -> str:
    from app.db import get_connection
    from app.decision import rules_engine
    from app.decision.database import ensure_decision_tables, get_decisions, get_mission_plan

    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        decisions = get_decisions(conn, survey_id)
        plan = get_mission_plan(conn, survey_id)
        sim_row = conn.execute(
            "SELECT detail_json FROM execution_logs WHERE survey_id=? AND event='summary' "
            "ORDER BY log_id DESC LIMIT 1", (survey_id,)).fetchone()
    finally:
        conn.close()
    sim = json.loads(sim_row[0]) if sim_row else None
    rules = rules_engine.load_rules()
    summary = rules_engine.summarise_decisions(decisions, rules) if decisions else None
    metrics = plan["metrics"] if plan else {}
    field_w = plan["field"]["width_m"] if plan else 10.0
    field_h = plan["field"]["height_m"] if plan else 10.0
    gt_eval = evaluate_against_ground_truth(survey_id)
    statuses = ({k: v.get("status") for k, v in (pipeline_result or {}).items()
                 if isinstance(v, dict) and k.startswith("stage_")})

    kpis = []
    if summary:
        kpis += [
            _kpi("Plants assessed", str(summary["total_plants"]),
                 f"{summary['plants_sprayed']} need spraying"),
            _kpi("Chemical saved", f"{summary['chemical_saved_pct']:.0f}%",
                 f"{summary['chemical_used_ml']:.0f} mL vs {summary['blanket_baseline_ml']:.0f} mL blanket"),
            _kpi("Cost saved", f"₹{summary['cost_saved_inr']:.2f}",
                 f"₹{summary['cost_actual_inr']:.2f} vs ₹{summary['cost_blanket_inr']:.2f}"),
        ]
    if metrics:
        kpis += [
            _kpi("Flight distance", f"{metrics['optimised_distance_m']:.1f} m",
                 f"{metrics['distance_saved_vs_blanket_pct']:.0f}% shorter than blanket pass"),
            _kpi("Mission time", f"{metrics['flight_time_s'] / 60:.1f} min",
                 f"{metrics['sorties']} sortie(s), battery {metrics['battery_used_pct']:.0f}%"),
        ]
    if gt_eval:
        kpis.append(_kpi("Demo accuracy", f"{(gt_eval['classification_accuracy_on_matched'] or 0) * 100:.1f}%",
                         f"detection recall {(gt_eval['detection_recall'] or 0) * 100:.0f}%"))

    rows = []
    gradcam_dir = OUTPUTS_ROOT / survey_id / "gradcam"
    for d in decisions:
        thumb = _img_data_uri(gradcam_dir / f"plant_{d['plant_id']:03d}.jpg")
        thumb_html = f'<img src="{thumb}" alt="Grad-CAM plant {d["plant_id"]}" class="thumb">' if thumb else "—"
        loc = (f"{d['field_x_m']:.2f}, {d['field_y_m']:.2f}" if d["is_mapped"] else "unmapped")
        conf = f"{d['confidence']:.2f}" if d["confidence"] is not None else "—"
        sev = (f"{d['severity_category']} ({d['severity_score']:.0%})"
               if d.get("severity_score") is not None and d.get("severity_category") else "—")
        color = LABEL_COLORS.get(d["disease_label"], "#868e96")
        rows.append(
            f"<tr><td>{d['plant_id']}</td>"
            f"<td><span class='dot' style='background:{color}'></span>{html.escape(d['disease_label'])}</td>"
            f"<td>{conf}</td><td>{html.escape(sev)}</td><td>{loc}</td>"
            f"<td class='act act-{d['action']}'>{ACTION_TEXT.get(d['action'], d['action'])}</td>"
            f"<td>{d['dosage_ml']:g}</td><td class='reason'>{html.escape(d['reason'])}</td>"
            f"<td>{thumb_html}</td></tr>")

    legend = "".join(f"<span><span class='dot' style='background:{c}'></span>{k.replace('_', ' ')}</span>"
                     for k, c in LABEL_COLORS.items() if k != "unclassified")
    stage_html = "".join(f"<li><code>{html.escape(k)}</code> <b class='st st-{html.escape(str(v))}'>"
                         f"{html.escape(str(v))}</b></li>" for k, v in statuses.items())
    gt_html = ""
    if gt_eval:
        gt_html = ("<section><h2>System-level check against the demo ground truth</h2>"
                   f"<p class='muted'>{html.escape(gt_eval['note'])}</p><table class='kv'>"
                   + "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{html.escape(str(v))}</td></tr>"
                             for k, v in gt_eval.items() if k != "note")
                   + "</table></section>")
    sim_html = ""
    if sim:
        sim_html = ("<section><h2>Software-in-the-loop flight simulation</h2><table class='kv'>"
                    + "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{html.escape(str(v))}</td></tr>"
                              for k, v in sim.items())
                    + "</table></section>")
    mission_files = sorted((OUTPUTS_ROOT / survey_id / "mission").glob("*")) \
        if (OUTPUTS_ROOT / survey_id / "mission").exists() else []
    files_html = "".join(f"<li><code>{html.escape(p.relative_to(REPO_ROOT).as_posix())}</code></li>"
                         for p in mission_files)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartSpray survey report</title>
<style>
:root {{ --bg:#f8f9fa; --card:#ffffff; --ink:#212529; --muted:#6c757d; --line:#dee2e6;
        --soil:#efe6d8; --grid:#d9cbb4; --accent:#2b8a3e; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141517; --card:#1f2124; --ink:#e9ecef; --muted:#adb5bd; --line:#343a40;
          --soil:#2b2620; --grid:#40382c; --accent:#69db7c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 12px; }}
.muted {{ color:var(--muted); }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:18px; margin-top:18px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:18px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.kpi-label {{ color:var(--muted); font-size:13px; }}
.kpi-value {{ font-size:26px; font-weight:650; margin:2px 0; font-variant-numeric:tabular-nums; }}
.kpi-sub {{ color:var(--muted); font-size:12px; }}
.fieldmap {{ width:100%; height:auto; }}
.soil {{ fill:var(--soil); }} .grid {{ stroke:var(--grid); stroke-width:1; }}
.axis {{ fill:var(--muted); font-size:11px; text-anchor:middle; }}
.pid {{ fill:var(--ink); font-size:9px; text-anchor:middle; }}
.route {{ fill:none; stroke-width:2; stroke-dasharray:6 4; opacity:.85; }}
.spray-ring {{ fill:none; stroke-width:2.5; }}
.depot {{ fill:var(--ink); }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:var(--muted); margin-top:8px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }}
th {{ color:var(--muted); font-weight:600; }}
td.reason {{ max-width:340px; color:var(--muted); }}
.act-spray {{ color:#c2255c; font-weight:650; }} .act-inspect {{ color:#e8590c; font-weight:600; }}
.thumb {{ width:56px; height:56px; object-fit:cover; border-radius:6px; }}
.kv th {{ width:40%; }}
code {{ overflow-wrap:anywhere; word-break:break-word; }}
.kv td {{ overflow-wrap:anywhere; }}
ul.stages {{ columns:2; padding-left:18px; }}
.st-success,.st-found {{ color:var(--accent); }} .st-skipped {{ color:#e8590c; }} .st-error {{ color:#e03131; }}
@media (max-width:640px) {{ ul.stages {{ columns:1; }} .kpi-value {{ font-size:22px; }} }}
</style></head>
<body><main>
<h1>SmartSpray — survey <code>{html.escape(survey_id)}</code></h1>
<div class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · rules {html.escape(rules.version)} · {html.escape(rules.status)}</div>
<div class="kpis">{''.join(kpis)}</div>
<section><h2>Field map and spray route</h2>
{_field_svg(decisions, plan, field_w, field_h)}
<div class="legend">{legend}<span>◯ ring = spray target</span><span>◆ = manual inspection</span><span>■ = depot</span><span>axes in metres</span></div>
{('<p class="muted">' + html.escape(plan.get('message') or '') + '</p>') if plan and plan.get('message') else ''}
</section>
<section><h2>Per-plant decisions</h2><div class="tablewrap"><table>
<thead><tr><th>Plant</th><th>Disease</th><th>Conf.</th><th>Severity</th><th>x, y (m)</th><th>Action</th><th>mL</th><th>Reason</th><th>Grad-CAM</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=9>No decisions yet.</td></tr>'}</tbody></table></div></section>
{gt_html}
{sim_html}
<section><h2>Pipeline stages</h2><ul class="stages">{stage_html or '<li>not run from the pipeline</li>'}</ul>
<h2>Mission files</h2><ul>{files_html or '<li>none</li>'}</ul></section>
</main></body></html>"""
    out_dir = OUTPUTS_ROOT / survey_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    summary_path = out_dir / "pipeline_result.json"
    if pipeline_result is not None:
        summary_path.write_text(json.dumps(pipeline_result, indent=2, default=str), encoding="utf-8")
    if gt_eval:
        (out_dir / "demo_ground_truth_evaluation.json").write_text(
            json.dumps(gt_eval, indent=2), encoding="utf-8")
    return path.relative_to(REPO_ROOT).as_posix()
