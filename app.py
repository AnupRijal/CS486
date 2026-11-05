"""
Flight Delay Prediction

    streamlit run app.py
"""

import airportsdata
import altair as alt
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

from features import SCHEDULE, CATEGORIES

st.set_page_config(page_title="Flight Delay Prediction",
                   page_icon="✈", layout="wide")

st.markdown("""
<style>
  .stApp { background:#f4f2ef; }
  .block-container { padding-top:1.4rem; max-width:1150px; }
  .hero { background:linear-gradient(135deg,#1b3a5c,#2d5a80);
          color:#fff; padding:1.7rem 2rem; border-radius:12px;
          margin-bottom:1.3rem; }
  .hero h1 { margin:0; font-size:2rem; font-weight:650; color:#fff; }
  .hero p { margin:.5rem 0 0; color:#c6d8e8; font-size:.95rem;
            max-width:64ch; line-height:1.55; }
  .card { background:#fff; border:1px solid #e4dfd8; border-radius:11px;
          padding:1.2rem 1.35rem; margin-bottom:.9rem; height:100%; }
  .card h4 { margin:0 0 .9rem; font-size:.76rem; font-weight:680;
             letter-spacing:.1em; text-transform:uppercase; color:#8a8178; }
  .big { font-size:3.7rem; font-weight:660; line-height:1; }
  .sub { font-size:1.08rem; font-weight:560; margin-top:.35rem; }
  .muted { color:#8a8178; font-size:.87rem; margin-top:.6rem;
           line-height:1.5; }
  .fact { display:flex; gap:.75rem; padding:.65rem 0;
          border-bottom:1px solid #f1ede7; }
  .fact:last-child { border-bottom:none; }
  .dot { width:9px; height:9px; border-radius:50%; margin-top:.45rem;
         flex-shrink:0; }
  .fact b { display:block; font-size:.93rem; color:#2c2924; font-weight:600;
            margin-bottom:.12rem; }
  .fact span { font-size:.855rem; color:#7d756c; line-height:1.45; }
  .verdict { padding:1rem 1.2rem; border-radius:10px; font-size:1.02rem;
             font-weight:560; }
</style>
""", unsafe_allow_html=True)

AIRPORTS = airportsdata.load("IATA")
CARRIERS = {"AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta",
            "F9": "Frontier", "G4": "Allegiant", "HA": "Hawaiian",
            "NK": "Spirit", "UA": "United", "WN": "Southwest",
            "9E": "Endeavor", "MQ": "Envoy", "OH": "PSA", "OO": "SkyWest",
            "YX": "Republic", "YV": "Mesa", "QX": "Horizon",
            "ZW": "Air Wisconsin", "C5": "CommuteAir", "G7": "GoJet",
            "MX": "Breeze"}
TONE = {"good": "#3d7a5c", "warn": "#c98b3a", "bad": "#c0564a",
        "none": "#a8a099"}


@st.cache_data
def get_data():
    return pd.read_parquet("outputs/app_data.parquet")


@st.cache_resource
def get_model():
    m = XGBClassifier()
    m.load_model("outputs/model_h24_A.json")
    return m


df = get_data()
BASE = df.DepDel15.mean()


def city(c):
    a = AIRPORTS.get(c)
    return f"{c} — {a['city']}" if a else c


def airline(c):
    return f"{c} — {CARRIERS.get(c, c)}"


def hhmm(t):
    t = int(t)
    return f"{t // 100:02d}:{t % 100:02d}"

def verdict(p):
    if p >= 0.45:
        return "High delay risk", "bad"
    if p >= 0.30:
        return "Raised delay risk", "warn"
    if p >= 0.18:
        return "Average delay risk", "none"
    return "Low delay risk", "good"


def facts_html(items):
    return "".join(
        f"<div class='fact'><div class='dot' style='background:{TONE[t]}'></div>"
        f"<div><b>{h}</b><span>{d}</span></div></div>" for t, h, d in items)


def weather_fact(v, w, codes):
    if pd.isna(v) and pd.isna(w):
        return "none", "Weather not reported", "This airport sent no observation."
    bits = []
    if pd.notna(v):
        bits.append(f"{v:.0f} mile visibility")
    if pd.notna(w):
        bits.append(f"{w:.0f} knot wind")
    d = ", ".join(bits)
    rough = isinstance(codes, str) and any(
        k in codes for k in ("SN", "TS", "FZ", "+RA"))
    if (pd.notna(v) and v < 3) or rough:
        return "bad", "Poor flying weather", \
            f"{d}. Airports space out departures in these conditions."
    if (pd.notna(v) and v < 6) or (pd.notna(w) and w > 20):
        return "warn", "Marginal weather", f"{d}. Enough to slow things a little."
    return "good", "Clear weather", f"{d}. No problem for departures."


def aircraft_fact(d):
    if pd.isna(d):
        return "none", "No earlier flight", \
            "This aircraft had not flown yet, so nothing could run late into it."
    if d >= 30:
        return "bad", f"Inbound aircraft ran {d:.0f} minutes late", \
            "A late plane makes its next flight late. This is the single " \
            "strongest warning sign the model has."
    if d >= 10:
        return "warn", f"Inbound aircraft ran {d:.0f} minutes late", \
            "Slightly behind. Time on the ground may absorb it."
    return "good", "Aircraft was running on schedule", \
        f"Its previous flight left {abs(d):.0f} minutes " \
        f"{'late' if d > 0 else 'early'}. Nothing to carry over."


def risk_chart(vals, labels):
    d = pd.DataFrame({"when": labels, "o": range(len(labels)), "risk": vals})
    bars = alt.Chart(d).mark_bar(height=28, cornerRadius=3).encode(
        y=alt.Y("when:N", sort=alt.EncodingSortField("o"), title=None),
        x=alt.X("risk:Q", title="Chance of a 15+ minute delay",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(0.65, max(vals) * 1.2)])),
        color=alt.condition(alt.datum.risk > BASE,
                            alt.value("#c0564a"), alt.value("#3d7a5c")))
    txt = bars.mark_text(align="left", dx=6, color="#5a544c",
                         fontSize=12).encode(text=alt.Text("risk:Q", format=".0%"))
    rule = alt.Chart(pd.DataFrame({"x": [BASE]})).mark_rule(
        strokeDash=[4, 4], color="#b5aea5").encode(x="x:Q")
    return (bars + txt + rule).properties(height=160)


st.markdown(
    "<div class='hero'><h1>Flight Delay Prediction</h1>"
    "<p>How the chance of a delayed departure is estimated at four moments "
    "before takeoff — a day ahead, half a day, six hours, and two hours. "
    "Built from 14 million US flights and 125 million airport weather "
    f"observations. On average {BASE:.0%} of flights leave 15 minutes late "
    "or more.</p></div>", unsafe_allow_html=True)

past, future = st.tabs(["  Look up a past flight  ", "  Plan a future flight  "])

# ------------------------------------------------------------------- past
with past:
    c1, c2, c3 = st.columns(3)
    o = c1.selectbox("Departing from", sorted(df.Origin.unique()), index=None,
                     placeholder="Choose an airport", format_func=city)
    sub = df[df.Origin == o] if o else df.iloc[0:0]
    d = c2.selectbox("Arriving at", sorted(sub.Dest.unique()), index=None,
                     placeholder="Choose an airport", format_func=city,
                     disabled=not o)
    sub = sub[sub.Dest == d] if d else sub.iloc[0:0]
    day = c3.selectbox("Date", sorted(sub.FlightDate.unique()), index=None,
                       placeholder="Choose a date", disabled=not d)

    if not day:
        st.info("Pick an airport, a destination and a date to see a flight.")
    else:
        sub = sub[sub.FlightDate == day].assign(label=[
            f"{airline(a)}  {int(n)}  ·  departs {hhmm(t)}"
            for a, n, t in zip(sub[sub.FlightDate == day].Reporting_Airline,
                               sub[sub.FlightDate == day]
                               .Flight_Number_Reporting_Airline,
                               sub[sub.FlightDate == day].CRSDepTime)
        ]).sort_values("CRSDepTime")

        pick = st.selectbox("Flight", sub.label, index=None,
                            placeholder="Choose a flight")
        if not pick:
            st.info(f"{len(sub)} flights on this route that day.")
        else:
            r = sub[sub.label == pick].iloc[0]
            label, tone = verdict(r.p02)

            st.write("")
            a, b = st.columns([1, 1.8])
            with a:
                st.markdown(
                                        f"<div class='card'><h4>Chance of delay · two hours out</h4>"
                    f"<div class='big' style='color:{TONE[tone]}'>"
                    f"{r.p02:.0%}</div>"
                    f"<div class='sub' style='color:{TONE[tone]}'>{label}</div>"
                    f"<div class='muted'>The typical US flight sits at "
                    f"{BASE:.0%}. Anything well above that means the model "
                    f"found a specific reason to worry.</div></div>",
                    unsafe_allow_html=True)
            with b:
                st.markdown("<div class='card'><h4>As departure approaches"
                            "</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    risk_chart([r.p24, r.p12, r.p06, r.p02],
                               ["24 hours before", "12 hours before",
                                "6 hours before", "2 hours before"]),
                    use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            e, f = st.columns([1.3, 1])
            with e:
                st.markdown(
                    "<div class='card'><h4>Why the model said that</h4>"
                    + facts_html([
                        weather_fact(r.o_vsby, r.o_sknt, r.o_wxcodes),
                        aircraft_fact(r.prev_dep_delay),
                        ("none", f"{r.Distance:,.0f} mile flight, "
                                 f"{int(r.CRSElapsedTime // 60)}h "
                                 f"{int(r.CRSElapsedTime % 60)}m scheduled",
                         "Longer flights and later departure times both carry "
                         "slightly more risk."),
                    ]) + "</div>", unsafe_allow_html=True)
            with f:
                late = r.DepDel15 == 1
                st.markdown(
                    f"<div class='card'><h4>What actually happened</h4>"
                    f"<div class='verdict' style='background:"
                    f"{'#fbeeec' if late else '#edf5f0'};color:"
                    f"{TONE['bad'] if late else TONE['good']}'>"
                    + (f"Left {r.DepDelay:.0f} minutes late" if late
                       else f"Left on time ({r.DepDelay:+.0f} min)")
                    + "</div></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------- future
with future:
    st.warning(
        "**No weather data exists for a future date.** Airport observations "
        "and aircraft positions only appear in the hours before departure. "
        "The estimate below is built from how this route, airline and time of "
        "day behaved across 2023–2026 — a seasonal and structural pattern, "
        "not a forecast. Treat it as a planning number. It will not improve "
        "as the date gets closer; only real weather and aircraft data can do "
        "that, and those arrive within about a day of departure.")

    g1, g2, g3 = st.columns(3)
    fo = g1.selectbox("Departing from", sorted(df.Origin.unique()), index=None,
                      placeholder="Choose an airport", format_func=city,
                      key="fo")
    rt = df[df.Origin == fo] if fo else df.iloc[0:0]
    fd = g2.selectbox("Arriving at", sorted(rt.Dest.unique()), index=None,
                      placeholder="Choose an airport", format_func=city,
                      disabled=not fo, key="fd")
    rt = rt[rt.Dest == fd] if fd else rt.iloc[0:0]
    when = g3.date_input("Date", value=None, min_value=pd.Timestamp("2026-08-01"),
                         disabled=not fd)

    if not (fd and when):
        st.info("Pick a route and a date.")
    else:
        h1, h2 = st.columns(2)
        car = h1.selectbox("Airline", sorted(rt.Reporting_Airline.unique()),
                           format_func=airline)
        dep = h2.time_input("Scheduled departure",
                            value=pd.Timestamp("08:00").time())

        x = pd.DataFrame([{
            "Month": when.month, "DayOfWeek": when.isoweekday(),
            "dep_minutes": dep.hour * 60 + dep.minute,
            "Reporting_Airline": car, "Origin": fo, "Dest": fd,
            "Distance": rt.Distance.median(),
            "CRSElapsedTime": rt.CRSElapsedTime.median()}])[SCHEDULE]
        for c, lv in CATEGORIES.items():
            if c in x:
                x[c] = pd.Categorical(x[c], categories=lv)

        p = float(get_model().predict_proba(x)[:, 1][0])
        label, tone = verdict(p)
        hist = rt[rt.FlightDate.astype(str).str[5:7] == f"{when.month:02d}"]

        st.write("")
        a, b = st.columns([1, 1.5])
        with a:
            st.markdown(
                f"<div class='card'><h4>Planning estimate</h4>"
                f"<div class='big' style='color:{TONE[tone]}'>{p:.0%}</div>"
                f"<div class='sub' style='color:{TONE[tone]}'>{label}</div>"
                f"<div class='muted'>Based on schedule and season only. "
                f"Average flight: {BASE:.0%}.</div></div>",
                unsafe_allow_html=True)
        with b:
            seen = (f"In past years, {hist.DepDel15.mean():.0%} of "
                    f"{len(hist):,} flights on this route in this month left "
                    f"late." if len(hist) > 30 else
                    "Too few past flights on this exact route and month to "
                    "quote a historical rate.")
            st.markdown(
                f"<div class='card'><h4>What this is built on</h4>"
                f"<div class='muted' style='margin-top:0'>{seen}<br><br>"
                f"Once the date is within about a day, weather at {fo} becomes "
                f"available and adds roughly a point of accuracy. In the final "
                f"two hours the inbound aircraft's status appears, and that is "
                f"worth far more — it carries most of the predictable signal. "
                f"Measuring that gap is what this project is about."
                f"</div></div>", unsafe_allow_html=True)