import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from deltiet_turbofan_engine import analyze_fleet, load_cmapss

st.set_page_config(
    page_title="Delt-IEt Turbofan Lab",
    page_icon="✈️",
    layout="wide",
)
st.title("✈️ Delt-IEt Turbofan Lab")
st.caption("Experimental environment for jet-engine degradation trajectory analysis — NASA CMAPSS")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Engine settings")
    st.markdown("**Mode:** compatible with original CMAPSS notebook")
    window_entropy = st.number_input(
        "Entropy window (W)", min_value=2, max_value=100, value=10,
        help="Number of time steps used to compute sliding Shannon entropy per sensor."
    )
    window_zscore = st.number_input(
        "Z-score window (H)", min_value=2, max_value=200, value=20,
        help="Lookback window for adaptive z-score normalisation."
    )
    percentile = st.number_input(
        "Threshold percentile (Θ)", min_value=50, max_value=99, value=95,
        help="Percentile of |Delt-IEt| used as detection threshold (computed retrospectively)."
    )
    st.divider()
    st.caption(
        "Parameters affect results. Record them when exporting "
        "to compare analyses across datasets."
    )

# ── Upload area ───────────────────────────────────────────────────────────────
st.markdown("""
Upload a **NASA CMAPSS** plain-text file (space-separated, no header).
The tool computes a multivariate Delt-IEt index for every engine trajectory
and reports the first detection cycle and remaining useful life (RUL) at alarm.
""")
st.warning(
    "Version 1.0 experimental. The result is a mathematical signal for investigation; "
    "it is not a certified diagnostic or a guaranteed failure prediction."
)

uploaded = st.file_uploader(
    "Choose a CMAPSS data file",
    type=["txt", "csv"],
    help="Files: train_FD001.txt … train_FD004.txt (or test sets). "
         "26 columns: unit, cycle, 3 op-conditions, 21 sensors.",
)

if uploaded is None:
    st.info(
        "Start by uploading a CMAPSS file. "
        "Each row must represent one time step of one engine "
        "with columns: unit · cycle · op1 · op2 · op3 · s1 … s21."
    )

    st.subheader("🚀 Don't have turbofan data? Use one of these free datasets")
    st.markdown("""
#### 1 · NASA CMAPSS — Turbofan Engine Degradation Simulation (FD001–FD004)

**👉 [Direct download — CMAPSS ZIP (≈ 2 MB)](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)**

Extract and upload any `train_FD00X.txt` — the tool recognises the format automatically.

| File | Engines | Op. conditions | Fault modes |
|------|---------|----------------|-------------|
| train_FD001.txt | 100 | 1 | 1 (HPC degradation) |
| train_FD002.txt | 260 | 6 | 1 |
| train_FD003.txt | 100 | 1 | 2 |
| train_FD004.txt | 249 | 6 | 2 |

> *A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set.*
> *NASA Ames PCoE.*

---

#### 2 · Kaggle mirror — same data, easy download

**👉 [NASA CMAPSS on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)**

---

#### 3 · Reference notebook — Delt-IEt applied to all 709 engines

**👉 [Kaggle: Delt-IEt — What is happening at cycle 8?](https://www.kaggle.com/code/ubirajaravasconcelos/delt-iet-what-is-happening-at-cycle-8)**

Results: 709 / 709 engines detected · zero failure labels · median first alarm 165 cycles before end-of-life.
""")

    st.subheader("Expected column layout")
    st.code(
        "unit  cycle  op1   op2   op3   s1    s2    …  s21\n"
        "1     1      -0.0  -0.0  100   518.6 641.8 …  23.4\n"
        "1     2      0.0   0.0   100   518.6 642.1 …  23.4\n"
        "…",
        language="text",
    )
    st.stop()

# ── Parse file ────────────────────────────────────────────────────────────────
try:
    df = load_cmapss(io.BytesIO(uploaded.getvalue()))
except Exception as exc:
    st.error(f"Could not parse the file: {exc}")
    st.stop()

n_engines = df['unit'].nunique()
n_cycles_total = len(df)
st.success(
    f"File recognised — **{n_engines} engines** · "
    f"**{n_cycles_total:,} time steps** · "
    f"columns: {list(df.columns[:6])} …"
)

# ── Run ───────────────────────────────────────────────────────────────────────
run_key = f"fleet_{uploaded.name}_{window_entropy}_{window_zscore}_{percentile}"
if st.button("▶ Run Delt-IEt on fleet", type="primary", use_container_width=True):
    with st.spinner(f"Analysing {n_engines} engine trajectories…"):
        fleet_df, per_engine = analyze_fleet(
            df,
            W=int(window_entropy),
            H=int(window_zscore),
            percentile=int(percentile),
            warmup=20,
        )
    st.session_state["fleet_df"] = fleet_df
    st.session_state["per_engine"] = per_engine
    st.session_state["run_key"] = run_key
    st.session_state["run_params"] = (
        int(window_entropy), int(window_zscore), int(percentile)
    )
    st.session_state["run_source"] = uploaded.name

# ── Results ───────────────────────────────────────────────────────────────────
if "fleet_df" in st.session_state:
    fleet_df: pd.DataFrame = st.session_state["fleet_df"]
    per_engine: dict = st.session_state["per_engine"]

    n_det = int(fleet_df["detected"].sum())
    n_total = len(fleet_df)
    det_rate = 100.0 * n_det / n_total if n_total > 0 else 0.0
    rul_detected = fleet_df.loc[fleet_df["detected"] == 1, "rul_at_first_alarm"].dropna()
    median_rul = float(rul_detected.median()) if len(rul_detected) else float("nan")
    earliest_rul = float(rul_detected.max()) if len(rul_detected) else float("nan")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Engines analysed", f"{n_total}")
    m2.metric("Engines detected", f"{n_det}")
    m3.metric("Detection rate", f"{det_rate:.1f}%")
    m4.metric("Median RUL at alarm", f"{median_rul:.0f} cycles" if not np.isnan(median_rul) else "—")
    m5.metric("Earliest alarm (max RUL)", f"{earliest_rul:.0f} cycles" if not np.isnan(earliest_rul) else "—")

    st.subheader("3. Results")
    tab1, tab2, tab3 = st.tabs(["Fleet Overview", "Engine Explorer", "Table"])

    # ── Tab 1: Fleet Overview ─────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**RUL distribution at first alarm**")
            if len(rul_detected) > 0:
                fig_hist = px.histogram(
                    rul_detected,
                    nbins=30,
                    labels={"value": "RUL at first alarm (cycles)", "count": "Engines"},
                    title=f"First-alarm RUL — {n_det} detected engines",
                    color_discrete_sequence=["#00C37A"],
                )
                fig_hist.update_layout(showlegend=False, margin=dict(t=40, b=20))
                fig_hist.add_vline(
                    x=median_rul, line_dash="dash", line_color="#FF6B35",
                    annotation_text=f"median {median_rul:.0f}",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No detections to display.")

        with col_b:
            st.markdown("**First-alarm cycle vs end-of-life cycle**")
            if len(rul_detected) > 0:
                det_rows = fleet_df[fleet_df["detected"] == 1].copy()
                fig_scatter = px.scatter(
                    det_rows,
                    x="max_cycle",
                    y="first_alarm_cycle",
                    hover_data=["unit", "rul_at_first_alarm"],
                    labels={
                        "max_cycle": "End-of-life cycle",
                        "first_alarm_cycle": "First alarm cycle",
                    },
                    title="Alarm lead time per engine",
                    color_discrete_sequence=["#00C37A"],
                )
                # Diagonal = alarm at end of life (zero lead)
                max_val = int(det_rows["max_cycle"].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines",
                    line=dict(dash="dot", color="gray", width=1),
                    name="Zero lead",
                    showlegend=True,
                ))
                fig_scatter.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No detections to display.")

        st.markdown(
            f"**{n_det} / {n_total}** engines produced at least one Delt-IEt event "
            f"before end of life. Threshold Θ computed retrospectively per engine "
            f"(percentile={int(st.session_state['run_params'][2])})."
        )

    # ── Tab 2: Engine Explorer ────────────────────────────────────────────────
    with tab2:
        engine_ids = sorted(per_engine.keys())
        selected = st.selectbox(
            "Select engine (unit number)",
            engine_ids,
            format_func=lambda u: (
                f"Engine {u} — detected at cycle "
                f"{per_engine[u]['first_alarm_cycle']} "
                f"(RUL {per_engine[u]['rul_at_alarm']})"
                if per_engine[u]['first_alarm_cycle'] is not None
                else f"Engine {u} — no detection"
            ),
        )
        if selected is not None:
            eng = per_engine[selected]
            cycles = eng["cycles"]
            combined = eng["combined"]
            theta = eng["theta"]
            first_alarm_cycle = eng["first_alarm_cycle"]
            first_alarm_idx = eng["first_alarm_idx"]

            fig_eng = go.Figure()
            # Main signal
            fig_eng.add_trace(go.Scatter(
                x=cycles, y=combined,
                mode="lines",
                name="|Delt-IEt| combined",
                line=dict(color="#4A9EFF", width=1.5),
            ))
            # Threshold
            fig_eng.add_hline(
                y=theta, line_dash="dash", line_color="#FF6B35",
                annotation_text=f"Θ = {theta:.4g}",
                annotation_position="bottom right",
            )
            # Alarm events
            event_mask = eng["events"] == 1
            if event_mask.any():
                fig_eng.add_trace(go.Scatter(
                    x=cycles[event_mask],
                    y=combined[event_mask],
                    mode="markers",
                    name="Events",
                    marker=dict(color="#FF6B35", size=4, symbol="circle"),
                ))
            # First alarm vertical line
            if first_alarm_cycle is not None:
                fig_eng.add_vline(
                    x=first_alarm_cycle, line_dash="dash", line_color="#00C37A",
                    annotation_text=f"First alarm — cycle {first_alarm_cycle}",
                    annotation_position="top left",
                )

            fig_eng.update_layout(
                title=f"Engine {selected} — {eng['max_cycle']} cycles · "
                      f"{eng['active_sensors']} active sensors",
                xaxis_title="Cycle",
                yaxis_title="|Delt-IEt| (combined max)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_eng, use_container_width=True)

            # Sensor entropy heatmap
            with st.expander("Sensor entropy heatmap (all active sensors)"):
                entropy_df = pd.DataFrame(
                    eng["entropy_matrix"].T,
                    columns=eng["active_sensor_names"],
                    index=cycles,
                )
                fig_heat = px.imshow(
                    entropy_df.T,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    labels=dict(x="Cycle", y="Sensor", color="Entropy"),
                    title=f"Shannon entropy per sensor — Engine {selected}",
                )
                if first_alarm_cycle is not None:
                    # Mark first alarm on heatmap
                    fig_heat.add_vline(
                        x=first_alarm_cycle - cycles[0],
                        line_dash="dash", line_color="white", line_width=1,
                    )
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Table ──────────────────────────────────────────────────────────
    with tab3:
        st.dataframe(
            fleet_df.rename(columns={
                "unit": "Engine",
                "max_cycle": "End-of-life cycle",
                "first_alarm_cycle": "First alarm cycle",
                "rul_at_first_alarm": "RUL at first alarm",
                "detected": "Detected",
                "active_sensors": "Active sensors",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download fleet results CSV",
            fleet_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="deltiet_turbofan_lab_results.csv",
            mime="text/csv",
        )
    with c2:
        W_p, H_p, pct_p = st.session_state["run_params"]
        report = (
            f"Delt-IEt Turbofan Lab\n"
            f"File: {st.session_state.get('run_source', '')}\n"
            f"W (entropy window) = {W_p}\n"
            f"H (z-score window) = {H_p}\n"
            f"Percentile = {pct_p}\n"
            f"Warmup = 20\n"
            f"Engines analysed = {n_total}\n"
            f"Engines detected = {n_det}\n"
            f"Detection rate = {det_rate:.2f}%\n"
            f"Median RUL at first alarm = {median_rul:.1f}\n"
            f"Earliest alarm (max RUL) = {earliest_rul:.1f}\n"
            "Threshold computed retrospectively per engine trajectory; "
            "does not represent online/real-time operation.\n"
        )
        st.download_button(
            "⬇ Download parameter report",
            report.encode("utf-8"),
            file_name="deltiet_turbofan_lab_report.txt",
            mime="text/plain",
        )

    # ── Interpretation ─────────────────────────────────────────────────────────
    st.subheader("4. How to interpret")
    st.markdown("""
- **Event** means the combined multivariate `|Delt-IEt|` exceeded the configured
  percentile threshold for that engine trajectory.
- The algorithm does **not** identify the physical degradation mechanism causing the event.
- Compare events against individual sensor readings, operating conditions, and
  independent health indicators to investigate the underlying signal.
- For scientific comparison, keep the same parameters (W, H, Θ) and record them.
""")
    st.caption(
        "Important limitation: this version computes the threshold retrospectively "
        "over each complete trajectory (percentile of the full series). "
        "It should not be presented as an online or real-time alert system."
    )
