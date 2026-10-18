"""
dashboard/streamlit_app.py

SmartSpray operator dashboard (Streamlit).

    streamlit run dashboard/streamlit_app.py

Pages (sidebar):
    Field overview  — KPIs, field map with spray route, per-plant decisions
    Explainability  — Grad-CAM heatmaps next to each plant crop
    Mission         — sorties, tank mixes, waypoint downloads, flight simulation
    Model card      — Module 3 test metrics, confusion matrix, training curves
    New survey      — upload images, register, run the whole pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.db import get_connection, init_db  # noqa: E402
from app.decision import rules_engine  # noqa: E402
from app.decision.database import get_decisions, get_mission_plan  # noqa: E402

EVAL_DIR = REPO_ROOT / "models" / "classification" / "eval"
OUTPUTS = REPO_ROOT / "data" / "outputs"
IMAGES = REPO_ROOT / "data" / "images"
LABEL_COLORS = {"healthy": "#2f9e44", "early_blight": "#e8590c",
                "late_blight": "#c2255c", "septoria_leaf_spot": "#1971c2"}

st.set_page_config(page_title="SmartSpray", page_icon="🌿", layout="wide")
init_db()


@st.cache_data(ttl=5)
def surveys() -> list[str]:
    conn = get_connection()
    try:
        return [r[0] for r in conn.execute(
            "SELECT survey_id FROM surveys ORDER BY created_at DESC").fetchall()]
    finally:
        conn.close()


def load(survey_id: str):
    conn = get_connection()
    try:
        decisions = get_decisions(conn, survey_id)
        plan = get_mission_plan(conn, survey_id)
        sim = conn.execute("SELECT detail_json FROM execution_logs WHERE survey_id=? "
                           "AND event='summary' ORDER BY log_id DESC LIMIT 1",
                           (survey_id,)).fetchone()
    finally:
        conn.close()
    return decisions, plan, json.loads(sim[0]) if sim else None


def field_figure(decisions, plan):
    import matplotlib.pyplot as plt

    w = plan["field"]["width_m"] if plan else 10
    h = plan["field"]["height_m"] if plan else 10
    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.set_facecolor("#efe6d8")
    if plan:
        for prog in plan.get("programmes", []):
            color = "#c2255c" if "late" in prog["treatment"] else "#e8590c"
            for s in prog["sorties"]:
                xs, ys = zip(*s["path"])
                ax.plot(xs, ys, "--", color=color, lw=1.5, alpha=0.8)
        ax.scatter([plan["config"]["depot_x_m"]], [plan["config"]["depot_y_m"]],
                   marker="s", s=90, color="black", label="depot")
    for d in decisions:
        if not d["is_mapped"]:
            continue
        c = LABEL_COLORS.get(d["disease_label"], "#868e96")
        marker = "D" if d["action"] == "inspect" else "o"
        ax.scatter(d["field_x_m"], d["field_y_m"], s=90, color=c, marker=marker,
                   edgecolors="black" if d["action"] == "spray" else "none", linewidths=2)
        ax.annotate(str(d["plant_id"]), (d["field_x_m"], d["field_y_m"] + 0.18),
                    ha="center", fontsize=7)
    for lbl, c in LABEL_COLORS.items():
        ax.scatter([], [], color=c, label=lbl.replace("_", " "))
    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, h + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8)
    fig.tight_layout()
    return fig


st.sidebar.title("🌿 SmartSpray")
page = st.sidebar.radio("Page", ["Field overview", "Explainability", "Mission",
                                 "Model card", "New survey"])
all_surveys = surveys()
survey_id = st.sidebar.selectbox("Survey", all_surveys) if all_surveys else None
st.sidebar.caption("Run `python scripts/run_demo.py` to create the demo survey.")

if page == "Field overview":
    st.header("Field overview")
    if not survey_id:
        st.info("No surveys yet — use **New survey** or run the demo script.")
    else:
        decisions, plan, sim = load(survey_id)
        if not decisions:
            st.warning("No treatment decisions yet for this survey — run the pipeline.")
        else:
            rules = rules_engine.load_rules()
            s = rules_engine.summarise_decisions(decisions, rules)
            c = st.columns(5)
            c[0].metric("Plants assessed", s["total_plants"])
            c[1].metric("Need spraying", s["plants_sprayed"])
            c[2].metric("Chemical saved", f"{s['chemical_saved_pct']:.0f}%",
                        f"{s['chemical_saved_ml']:.0f} mL")
            c[3].metric("Cost saved", f"₹{s['cost_saved_inr']:.2f}")
            if plan:
                c[4].metric("Flight distance", f"{plan['metrics']['optimised_distance_m']:.1f} m",
                            f"-{plan['metrics']['distance_saved_vs_blanket_pct']:.0f}% vs blanket",
                            delta_color="inverse")
            left, right = st.columns([3, 2])
            with left:
                st.pyplot(field_figure(decisions, plan))
            with right:
                st.subheader("Actions")
                st.bar_chart(pd.Series(s["actions"]))
                st.subheader("Plants by label")
                st.bar_chart(pd.Series(s["plants_by_label"]))
            df = pd.DataFrame(decisions)[["plant_id", "disease_label", "confidence",
                                          "severity_category", "severity_score",
                                          "field_x_m", "field_y_m", "action",
                                          "dosage_ml", "treatment", "reason"]]
            st.subheader("Per-plant decisions")
            st.dataframe(df, width="stretch", hide_index=True)
            report = OUTPUTS / survey_id / "report.html"
            if report.exists():
                st.download_button("Download HTML report", report.read_bytes(),
                                   file_name=f"{survey_id}_report.html", mime="text/html")

elif page == "Explainability":
    st.header("Why did the model say that? (Grad-CAM)")
    if survey_id:
        decisions, _, _ = load(survey_id)
        gdir = OUTPUTS / survey_id / "gradcam"
        imgs = sorted(gdir.glob("plant_*.jpg")) if gdir.exists() else []
        if not imgs:
            st.info("No Grad-CAM images — run the pipeline with Grad-CAM enabled.")
        by_id = {d["plant_id"]: d for d in decisions}
        filt = st.multiselect("Show labels", list(LABEL_COLORS), default=list(LABEL_COLORS))
        cols = st.columns(6)
        shown = 0
        for p in imgs:
            pid = int(p.stem.split("_")[1])
            d = by_id.get(pid)
            if d and d["disease_label"] not in filt:
                continue
            cap = (f"#{pid} {d['disease_label']} ({d['confidence']:.2f}) → {d['action']}"
                   if d else f"#{pid}")
            cols[shown % 6].image(str(p), caption=cap, width="stretch")
            shown += 1

elif page == "Mission":
    st.header("Spray mission")
    if survey_id:
        _, plan, sim = load(survey_id)
        if not plan:
            st.info("No mission plan yet.")
        else:
            m = plan["metrics"]
            c = st.columns(4)
            c[0].metric("Sorties", m["sorties"], f"{m['tank_refills']} refill(s)")
            c[1].metric("Distance", f"{m['optimised_distance_m']:.1f} m",
                        f"{m['route_improvement_vs_id_order_pct']:.0f}% shorter than id order")
            c[2].metric("Mission time", f"{m['flight_time_s'] / 60:.1f} min",
                        f"blanket {m['blanket_flight_time_s'] / 60:.1f} min", delta_color="off")
            c[3].metric("Battery", f"{m['battery_used_pct']:.0f}%")
            st.subheader("Tank mixes")
            st.dataframe(pd.DataFrame(plan["mix_plan"]), hide_index=True,
                         width="stretch")
            rows = [{k: s[k] for k in ("sortie", "treatment", "load_ml", "distance_m",
                                       "flight_time_s")} | {"plants": s["plant_ids"]}
                    for p in plan["programmes"] for s in p["sorties"]]
            st.subheader("Sorties")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            mdir = OUTPUTS / survey_id / "mission"
            for f in sorted(mdir.glob("*")) if mdir.exists() else []:
                st.download_button(f"Download {f.name}", f.read_bytes(), file_name=f.name,
                                   key=f.name)
            if st.button("Run flight simulation"):
                from simulator.executor_simulator import run
                sim = run(survey_id)
            if sim:
                st.subheader("Software-in-the-loop simulation")
                st.json(sim)

elif page == "Model card":
    st.header("Module 3 — disease classifier")
    metrics_path = EVAL_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("Run scripts/m3_train_classifier.py to produce the evaluation.")
    else:
        mt = json.loads(metrics_path.read_text(encoding="utf-8"))
        c = st.columns(4)
        c[0].metric("Test accuracy", f"{mt['test_accuracy'] * 100:.2f}%")
        c[1].metric("Macro F1", f"{mt['macro_f1']:.4f}")
        c[2].metric("Test images", mt["total_test_samples"])
        c[3].metric("Errors", mt["incorrect"])
        st.caption(f"Split: {mt['counts']} · seed 42 · manifest `{mt['split_manifest']}`")
        st.dataframe(pd.DataFrame(mt["per_class"]).T, width="stretch")
        cols = st.columns(2)
        for i, name in enumerate(["confusion_matrix.png", "training_curves.png",
                                  "reliability_diagram.png", "gradcam_examples.png"]):
            if (EVAL_DIR / name).exists():
                cols[i % 2].image(str(EVAL_DIR / name), width="stretch")

elif page == "New survey":
    st.header("Register a survey and run the pipeline")
    new_id = st.text_input("Survey ID", "survey_001")
    w = st.number_input("Field width (m)", 1.0, 1000.0, 10.0)
    h = st.number_input("Field height (m)", 1.0, 1000.0, 10.0)
    files = st.file_uploader("Survey images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    detector = st.selectbox("Detector", ["auto", "yolo", "classical"])
    if st.button("Register + run", type="primary"):
        if not new_id.strip() or "/" in new_id:
            st.error("Enter a plain survey id.")
        else:
            folder = IMAGES / new_id.strip()
            folder.mkdir(parents=True, exist_ok=True)
            for f in files or []:
                (folder / Path(f.name).name).write_bytes(f.getbuffer())
            from app.field_survey.service import register_survey
            from app.pipeline import run_pipeline, stage_statuses
            try:
                if new_id not in surveys():
                    st.write(register_survey(new_id.strip(), field_width_m=w, field_height_m=h))
                with st.spinner("Running Modules 1-9…"):
                    result = run_pipeline(new_id.strip(), detector, True, True, True)
                st.success("Done")
                st.json(stage_statuses(result))
                surveys.clear()
            except Exception as exc:  # show the message instead of a stack trace
                st.error(str(exc))
