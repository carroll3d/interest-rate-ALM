
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from vasicek import vasicek_paths, vasicek_analytic_mean_var
from cir import cir_paths, cir_analytic_mean_var

st.set_page_config(page_title="Short-Rate Models: Vasicek & CIR", page_icon="📈", layout="wide")
st.title("📈 Short-Rate Model Visualizer — Vasicek & Cox–Ingersoll–Ross (CIR)")

# Model selection
model = st.selectbox("Choose model", ["Vasicek", "Cox–Ingersoll–Ross (CIR)"], index=0)

with st.sidebar:
    st.header("Common Parameters")
    T = st.number_input("Horizon T (years)", value=2.0, step=0.5, min_value=0.1)
    dt = st.number_input("Time step Δt (years)", value=0.01, step=0.005, min_value=0.001, format="%.5f")
    n_paths = st.slider("Number of simulated paths", 1, 500, 15)
    seed_on = st.checkbox("Set random seed", value=True)
    seed = st.number_input("Seed (if checked)", value=42, step=1) if seed_on else None

    if model == "Vasicek":
        st.header("Vasicek Params")
        r0 = st.number_input("Initial short rate r₀", value=0.05, step=0.005, format="%.5f", min_value=0.0)
        a = st.number_input("Speed of mean reversion a", value=0.10, step=0.01, format="%.5f", min_value=0.0)
        b = st.number_input("Long-term mean b", value=0.05, step=0.005, format="%.5f", min_value=0.0)
        sigma = st.number_input("Volatility σ", value=0.02, step=0.005, format="%.5f", min_value=0.0)
    else:
        st.header("CIR Params")
        r0 = st.number_input("Initial short rate r₀", value=0.05, step=0.005, format="%.5f", min_value=0.0)
        a = st.number_input("Speed of mean reversion κ", value=0.50, step=0.05, format="%.5f", min_value=0.0)
        b = st.number_input("Long-term mean θ", value=0.05, step=0.005, format="%.5f", min_value=0.0)
        sigma = st.number_input("Volatility σ", value=0.10, step=0.01, format="%.5f", min_value=0.0)

# Simulate & compute analytics
if model == "Vasicek":
    t, paths = vasicek_paths(r0, a, b, sigma, T, dt, n_paths, seed=int(seed) if seed_on else None)
    mean_t, var_t = vasicek_analytic_mean_var(r0, a, b, sigma, t)
    model_name = "Vasicek"
else:
    t, paths = cir_paths(r0, a, b, sigma, T, dt, n_paths, seed=int(seed) if seed_on else None)
    mean_t, var_t = cir_analytic_mean_var(r0, a, b, sigma, t)
    model_name = "CIR"

std_t = np.sqrt(var_t) if np.all(np.isfinite(var_t)) else None

# DataFrames for charts
df_paths = pd.DataFrame(paths, columns=[f"path_{i+1}" for i in range(paths.shape[1])])
df_paths.insert(0, "t", t)
df_summary = pd.DataFrame({"t": t, "analytic_mean": mean_t})
if std_t is not None:
    df_summary["analytic_plus_1sd"] = mean_t + std_t
    df_summary["analytic_minus_1sd"] = mean_t - std_t

st.markdown(f"""
This app simulates **{model_name}** short-rate paths.

**SDEs**  
- Vasicek:  \(\mathrm{{d}}r_t = a (b - r_t) \, \mathrm{{d}}t + \sigma \, \mathrm{{d}}W_t\)  
- CIR:      \(\mathrm{{d}}r_t = \kappa (\theta - r_t) \, \mathrm{{d}}t + \sigma \sqrt{{r_t}} \, \mathrm{{d}}W_t\)

It overlays the **analytic mean** and, when available, a **±1σ band**.
""")

# Chart: multiple paths with analytic mean/band
st.subheader(f"Simulated short-rate paths — {model_name}")
base = alt.Chart(df_paths).transform_fold(
    fold=[c for c in df_paths.columns if c != "t"],
    as_=["series", "value"]
).mark_line(opacity=0.6).encode(
    x=alt.X("t:Q", title="Time (years)"),
    y=alt.Y("value:Q", title="Short rate"),
    detail="series:N"
)

if std_t is not None:
    band = alt.Chart(df_summary).mark_area(opacity=0.2).encode(
        x="t:Q",
        y="analytic_minus_1sd:Q",
        y2="analytic_plus_1sd:Q"
    )
    mean_line = alt.Chart(df_summary).mark_line(strokeWidth=2).encode(x="t:Q", y="analytic_mean:Q")
    chart = band + mean_line + base
else:
    mean_line = alt.Chart(df_summary).mark_line(strokeWidth=2).encode(x="t:Q", y="analytic_mean:Q")
    chart = mean_line + base

st.altair_chart(chart, use_container_width=True)

# Histogram at final horizon
st.subheader("Distribution at horizon T")
final_rates = df_paths.drop(columns=["t"]).iloc[-1].values
df_final = pd.DataFrame({"rate": final_rates})
hist = alt.Chart(df_final).mark_bar(opacity=0.8).encode(
    x=alt.X("rate:Q", bin=alt.Bin(maxbins=40), title="r(T)"),
    y=alt.Y("count()", title="Count")
)
st.altair_chart(hist, use_container_width=True)

# Table preview
with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df_paths.head(10))

# Downloads
csv = df_paths.to_csv(index=False).encode()
st.download_button("Download simulated paths as CSV", csv, file_name="short_rate_paths.csv", mime="text/csv")
