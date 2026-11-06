"""
Negative-price early-warning dashboard (Germany-Luxembourg day-ahead market).

Run from the repository root:
    pip install streamlit
    streamlit run app.py

Tabs
    Forecast         tomorrow's hourly probability of a negative price (live, or the latest saved one)
    What-if          you supply your own weather expectation (relative to yesterday); the model answers
    Replay           any past day: what the tool would have said, next to what really happened
    Track record     the append-only prediction log scored against realised prices
    Data timing      when the inputs are actually published (the evidence behind the 'before the auction' claim)
    About            method, validated performance, limitations
"""
import asyncio
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

if sys.platform == "win32":          # avoids event-loop errors when Streamlit runs on Windows
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:  # noqa: BLE001  (policy API changes in newer Python versions)
        pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import dashboard_logic as dl  # noqa: E402

st.set_page_config(page_title="Negative-price early warning", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: the decision setting shared by all tabs
# ---------------------------------------------------------------------------
st.sidebar.header("Decision setting")
ratio = st.sidebar.select_slider(
    "How much worse is a MISSED negative hour than a FALSE ALARM?",
    options=[0.25, 0.5, 1, 2, 4, 8], value=1,
    format_func=lambda r: f"{r:g}x",
)
threshold = dl.cost_ratio_to_threshold(ratio)
st.sidebar.markdown(
    f"An hour is **flagged** when its probability is at least **{threshold:.0%}**.\n\n"
    "1x means a miss and a false alarm cost the same (the setting the model was validated with). "
    "Larger values flag more hours. The rule is only optimal if the probabilities are well calibrated; "
    "check the Track record tab."
)
st.sidebar.caption("Not financial or trading advice. Research prototype.")


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------
def show_status(record):
    status = record["status"]
    level = status["level"]
    if level == "ok":
        st.success("Inputs complete for this day.")
    elif level == "minor_gaps":
        st.info("Inputs mostly complete (small gaps were filled from earlier values).")
    else:
        st.warning("Data quality is degraded for this forecast. Read the notes below.")
    for note in status["notes"]:
        st.caption(note)
    if record.get("source") == "api" and record.get("before_gate_closure") is False:
        st.warning("Generated after 12:00 Berlin time on the day before delivery: the real prices may already be "
                   "public, so this is NOT a prospective forecast.")


def probability_chart(df, thr):
    plot = df[["hour_label", "probability", "flag"]].copy()
    try:
        import altair as alt

        order = plot["hour_label"].tolist()
        bars = alt.Chart(plot).mark_bar().encode(
            x=alt.X("hour_label:N", sort=order, title="Hour of day (Berlin time)"),
            y=alt.Y("probability:Q", title="Probability of a negative price", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.flag, alt.value("#d1495b"), alt.value("#9aa9b8")),
            tooltip=["hour_label", alt.Tooltip("probability:Q", format=".2f")],
        )
        rule = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(strokeDash=[5, 5]).encode(y="threshold:Q")
        st.altair_chart((bars + rule).properties(height=300, width="container"))
    except Exception:  # noqa: BLE001  (fall back to a plain chart)
        st.line_chart(plot.set_index("hour_label").assign(threshold=thr)[["probability", "threshold"]])


def show_forecast(record, thr, prices=None):
    df = dl.hours_frame(record, thr)
    if prices is not None:
        df = dl.attach_actuals(df, prices)
    s = dl.summarize(df)

    show_status(record)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected negative hours", f"{s['expected_negative_hours']:.1f}")
    c2.metric("Highest hourly probability", f"{s['max_probability']:.0%}")
    c3.metric("Hours flagged", s["hours_flagged"])
    c4.metric("Decision threshold", f"{thr:.0%}")

    if s["windows"]:
        txt = "; ".join(f"{w['start']}-{w['end']} (peak {w['max_probability']:.0%})" for w in s["windows"])
        st.markdown(f"**Flagged windows (Berlin time):** {txt}")
    else:
        st.markdown("**Flagged windows:** none at this threshold")
    if s["hours_without_forecast"]:
        st.caption(f"{s['hours_without_forecast']} hour(s) have no forecast (extra hour of a 25-hour clock-change day).")

    probability_chart(df, thr)

    if prices is not None:
        sc = dl.day_scorecard(df)
        st.markdown(
            f"**What actually happened:** {sc['negative_hours']} negative hours. At this threshold the tool flagged "
            f"{sc['flagged_hours']}: caught {sc['caught']}, false alarms {sc['false_alarms']}, missed {sc['missed']}."
        )
        st.caption("Actual day-ahead price per hour (EUR/MWh)")
        st.bar_chart(df.set_index("hour_label")["actual_price"])

    cols = {"hour_label": "Hour", "probability": "Probability", "flag": "Flagged", "load_mw": "Load (MW)",
            "solar_mw": "Solar (MW)", "wind_mw": "Wind (MW)", "residual_load_mw": "Residual load (MW)",
            "coupled_residual_mw": "Coupled residual (MW)"}
    if prices is not None:
        cols["actual_price"] = "Actual price"
    table = df[list(cols)].rename(columns=cols)
    st.dataframe(table, hide_index=True)

    with st.expander("Why? Main drivers per hour (positive = pushes toward a negative price)"):
        st.dataframe(df[["hour_label", "probability", "drivers"]].rename(
            columns={"hour_label": "Hour", "probability": "Probability", "drivers": "Top drivers"}), hide_index=True)
    with st.expander("Model details"):
        m = record["model"]
        st.markdown(
            f"- Type: {m['type']} (regularisation C = {m['C']})\n"
            f"- Trained as of: {m['trained_as_of']} on the previous {m['window_days']} days "
            f"({m['n_train']:,} hours, {m['n_train_negative']:,} negative)\n"
            f"- Generated at: {record['generated_at_utc']}\n"
            f"- Neighbour series used: {len(record['status']['neighbour_series_used'])}")


@st.cache_data(show_spinner=False)
def get_prices():
    from data_loader import get_energy_data

    return get_energy_data()["price"]


@st.cache_data(show_spinner=False)
def run_replay(day):
    import predict_tomorrow as pt

    return pt.forecast_day(day, source="replay")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Negative-price early warning, Germany-Luxembourg")
st.caption("Hourly probability that the day-ahead electricity price is below zero. "
           "Useful for planning flexible electricity use, if the inputs are available early enough "
           "(see the Data timing tab).")

tab_live, tab_whatif, tab_replay, tab_track, tab_timing, tab_about = st.tabs(
    ["Forecast", "What-if", "Replay a past day", "Track record", "Data timing", "About"])

# ---- Forecast ---------------------------------------------------------------
with tab_live:
    tomorrow = (pd.Timestamp.now(tz=dl.LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date().isoformat()
    st.subheader("Live forecast")
    if st.button(f"Run live forecast for {tomorrow}"):
        with st.spinner("Downloading forecasts from ENTSO-E (this can take 1-3 minutes)..."):
            try:
                import predict_tomorrow as pt

                rec = pt.forecast_day(tomorrow, source="api", refresh=True)
                pt.save_record(rec)
                pt.append_log(rec)
                st.success("Forecast created and logged.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not produce the forecast: {e}")

    saved = dl.list_live_forecasts()
    if not saved:
        st.info("No live forecast saved yet. Click the button above, or run "
                "`python src/predict_tomorrow.py` in a terminal. To see the tool working right away, "
                "open the **Replay** tab.")
    else:
        choice = st.selectbox("Saved live forecasts (newest first)", [d for d, _ in saved])
        record = dl.load_record(dict(saved)[choice])
        st.markdown(f"### Delivery day {record['target_date']}")
        show_forecast(record, threshold)

# ---- What-if ----------------------------------------------------------------
with tab_whatif:
    st.subheader("What-if explorer")
    st.markdown(
        "Before noon the real wind and solar forecasts for tomorrow are usually not published yet. "
        "Here **you** supply your own expectation, for example from a weather app: the model starts from "
        "**yesterday's published forecasts** and scales them. 'Solar 1.3x' means you expect 30% more solar "
        "than yesterday. The result is a **scenario, not a forecast**, and nothing is logged.")
    wi_mode = st.radio("Day to explore", ["Tomorrow (live data)", "A past day (replay)"])
    wi_replay = wi_mode.startswith("A past")
    if wi_replay:
        try:
            wi_max = get_prices().dropna().index.max().tz_convert(dl.LOCAL_TZ).date()
        except Exception:  # noqa: BLE001
            wi_max = date(2026, 9, 30)
        wi_day = st.date_input("Past day", value=date(2025, 6, 15), min_value=date(2020, 3, 1),
                               max_value=wi_max, key="wi_day").isoformat()
    else:
        wi_day = (pd.Timestamp.now(tz=dl.LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date().isoformat()
    st.caption(f"Delivery day: {wi_day}")
    wc1, wc2 = st.columns(2)
    f_solar = wc1.slider("German solar vs yesterday", 0.0, 2.0, 1.0, 0.05)
    f_wind = wc1.slider("German wind vs yesterday", 0.0, 2.0, 1.0, 0.05)
    f_load = wc1.slider("German electricity demand vs yesterday", 0.8, 1.2, 1.0, 0.01)
    f_nb_solar = wc2.slider("Neighbours' solar vs yesterday", 0.0, 2.0, 1.0, 0.05)
    f_nb_wind = wc2.slider("Neighbours' wind vs yesterday", 0.0, 2.0, 1.0, 0.05)
    if st.button("Compute scenario"):
        with st.spinner("Computing ..."):
            try:
                import predict_tomorrow as pt

                rec = pt.forecast_day(
                    wi_day, source="replay" if wi_replay else "api",
                    scenario={"solar": f_solar, "wind": f_wind, "load": f_load,
                              "nb_solar": f_nb_solar, "nb_wind": f_nb_wind})
                st.session_state["whatif"] = {"record": rec, "replay": wi_replay}
            except Exception as e:  # noqa: BLE001
                st.session_state["whatif"] = None
                st.error(f"Could not compute the scenario: {e}")
    wi = st.session_state.get("whatif")
    if wi:
        st.markdown(f"### Scenario for {wi['record']['target_date']}")
        try:
            wi_prices = get_prices() if wi["replay"] else None
        except Exception:  # noqa: BLE001
            wi_prices = None
        show_forecast(wi["record"], threshold, prices=wi_prices)

# ---- Replay -----------------------------------------------------------------
with tab_replay:
    st.subheader("Replay a past day")
    st.caption("The model only uses information from before the chosen day (previous 365 days for training; "
               "forecasts and earlier prices for the day itself). Actual prices are shown only for comparison.")
    try:
        prices = get_prices()
        last_day = prices.dropna().index.max().tz_convert(dl.LOCAL_TZ).date()
        examples = dl.negative_hours_by_day(prices)
        labels = ["Custom date (use the picker below)"] + [f"{d} ({n} negative hours)" for d, n in examples]
        pick = st.selectbox("Example days with many negative hours", labels)
        custom = st.date_input("Custom date", value=date(2025, 6, 15), min_value=date(2020, 3, 1), max_value=last_day)
        day = custom.isoformat() if pick == labels[0] else examples[labels.index(pick) - 1][0]
        with st.spinner(f"Building the forecast for {day} ..."):
            rec = run_replay(day)
        st.markdown(f"### {day}")
        show_forecast(rec, threshold, prices=prices)
    except Exception as e:  # noqa: BLE001
        st.error(f"Replay is not available: {e}")

# ---- Track record -----------------------------------------------------------
with tab_track:
    st.subheader("Track record of live forecasts")
    st.caption("The log is append-only: a logged hour is never overwritten. Only forecasts generated BEFORE 12:00 "
               "Berlin time on the day before count as truly prospective.")
    log = dl.load_log()
    if log is None:
        st.info("No live forecasts logged yet. Each live forecast adds 24 rows.")
    else:
        if st.button("Fetch realised prices for finished days"):
            with st.spinner("Downloading realised prices ..."):
                try:
                    import predict_tomorrow as pt

                    pt.score_log(threshold)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not update outcomes: {e}")
            st.rerun()
        st.dataframe(dl.log_overview(log), hide_index=True)
        for label, only in [("All logged hours", False), ("Prospective only (before gate closure)", True)]:
            m = dl.log_metrics(log, threshold, only)
            st.markdown(f"**{label}**")
            if m is None:
                st.caption("No finished days with published prices yet.")
                continue
            cols = st.columns(5)
            cols[0].metric("Days / hours", f"{m['days']} / {m['hours']}")
            cols[1].metric("Negative hours", m["negative_hours"])
            cols[2].metric("Precision", f"{m['precision']:.2f}")
            cols[3].metric("Recall", f"{m['recall']:.2f}")
            cols[4].metric("Brier score", f"{m['brier']:.3f}")
        cal = dl.calibration_table(log, prospective_only=False)
        if cal is not None:
            st.markdown("**Calibration:** does a '70%' hour turn out negative about 70% of the time?")
            st.dataframe(cal.reset_index().rename(columns={"index": "probability band"}), hide_index=True)
            st.line_chart(cal[["mean_predicted", "observed_rate"]])
            st.caption("Few days means noisy numbers; winter has very few negative hours.")

# ---- Data timing ------------------------------------------------------------
with tab_timing:
    st.subheader("When are the inputs published?")
    st.markdown(
        "The day-ahead auction closes at **12:00 Berlin time** and its prices appear shortly afterwards. "
        "A forecast is only useful before that if all inputs (German load, wind, solar and the neighbours' "
        "forecasts) are already published. Take readings with "
        "`python src/predict_tomorrow.py --check-availability` (around 10:30, 13:00, 17:00 and 19:00 Berlin "
        "time; `--schedule` prints today's times in Berlin and Cairo time).")
    av = dl.load_availability()
    if av is None:
        st.info("No readings yet.")
    else:
        summ = dl.availability_summary(av)
        head, details = dl.timing_verdict(summ)
        st.markdown(f"**Reading so far:** {head}")
        for line in details:
            st.caption(line)
        st.markdown("**When does each input appear?** (Berlin time, day before delivery)")
        st.dataframe(dl.input_timing(summ), hide_index=True)
        st.caption("The true publication time lies between the two columns. 'Varies by day = yes' means the "
                   "order is reversed on some days, so the publication time is not constant.")
        show = summ[["checked_berlin", "target_date", "German_load", "German_wind_solar", "Neighbours",
                     "All_inputs_complete", "Tomorrows_price_public", "hours_before_gate_closure"]]
        st.dataframe(show.rename(columns={
            "checked_berlin": "Checked (Berlin)", "target_date": "Delivery day", "German_load": "German load",
            "German_wind_solar": "German wind+solar", "All_inputs_complete": "All inputs complete",
            "Tomorrows_price_public": "Price already public",
            "hours_before_gate_closure": "Hours before 12:00 gate"}), hide_index=True)
        st.caption("Series that were never published in any reading (for example a Czech wind forecast) "
                   "are ignored, because the model never used them.")

# ---- About ------------------------------------------------------------------
with tab_about:
    st.subheader("Method")
    st.markdown(
        "- **Question:** will the day-ahead price in Germany-Luxembourg be below zero in a given hour?\n"
        "- **Inputs:** day-ahead forecasts of load, wind and solar for Germany and nine neighbouring countries, "
        "calendar information, and prices up to the previous day. Prices of the delivery day are never used.\n"
        "- **Model:** logistic regression, refitted monthly on the previous 365 days, because the relationship "
        "between renewables and negative prices has shifted over time.\n"
        "- **Output:** a probability per hour. The sidebar turns your miss-versus-false-alarm costs into a cut-off.")
    st.subheader("Validated performance (test period, Jan 2025 to Sep 2026)")
    res = dl.validated_results()
    if res is None:
        st.caption("Run `python src/final.py` to produce results/final/test_results.csv.")
    else:
        st.dataframe(res, hide_index=True)
        st.caption("Rows: German-only rule (static and rolling), coupled-market rule, and the models. "
                   "PR-AUC measures how well hours are ranked; Brier measures probability quality (lower is better). "
                   "These numbers use forecasts as stored on ENTSO-E, which may be published later than the "
                   "auction deadline (see Data timing).")
    st.subheader("Limitations")
    st.markdown(
        "- **Timing:** if wind and solar forecasts are published after 12:00, the tool cannot warn before the auction.\n"
        "- **Shallow negatives:** many negative hours are barely below zero; those are the hardest to call.\n"
        "- **One market, one model:** trained on Germany-Luxembourg since 2019; structural changes can degrade it, "
        "which is why it is refitted monthly.\n"
        "- **15-minute products** since October 2025 are averaged to hourly prices.\n"
        "- **What-if results are scenarios,** driven by the factors you choose; they are not forecasts.\n"
        "- **Not advice:** a research prototype, not a trading or purchasing recommendation.")