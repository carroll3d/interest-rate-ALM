
    import io
    import numpy as np
    import pandas as pd
    import streamlit as st
    import altair as alt

    from vasicek import vasicek_paths, vasicek_analytic_mean_var

    st.set_page_config(page_title="Vasicek Model Visualizer", page_icon="📈", layout="wide")

    st.title("📈 Vasicek Short-Rate Model — Interactive Visualizer")

    with st.sidebar:
        st.header("Parameters")
        r0 = st.number_input("Initial short rate r₀", value=0.05, step=0.005, format="%.5f")
        a = st.number_input("Speed of mean reversion a", value=0.10, step=0.01, format="%.5f", min_value=0.0)
        b = st.number_input("Long-term mean b", value=0.05, step=0.005, format="%.5f")
        sigma = st.number_input("Volatility σ", value=0.02, step=0.005, format="%.5f", min_value=0.0)
        T = st.number_input("Horizon T (years)", value=2.0, step=0.5, min_value=0.1)
        dt = st.number_input("Time step Δt (years)", value=0.01, step=0.005, min_value=0.001, format="%.5f")
        n_paths = st.slider("Number of simulated paths", 1, 500, 50)
        seed_on = st.checkbox("Set random seed", value=True)
        seed = st.number_input("Seed (if checked)", value=42, step=1) if seed_on else None

    # Simulate
    t, paths = vasicek_paths(r0, a, b, sigma, T, dt, n_paths, seed=int(seed) if seed_on else None)
    mean_t, var_t = vasicek_analytic_mean_var(r0, a, b, sigma, t)
    std_t = np.sqrt(var_t)

    # DataFrames for charts
    df_paths = pd.DataFrame(paths, columns=[f"path_{i+1}" for i in range(paths.shape[1])])
    df_paths.insert(0, "t", t)
    df_summary = pd.DataFrame({"t": t, "analytic_mean": mean_t, "analytic_plus_1sd": mean_t + std_t, "analytic_minus_1sd": mean_t - std_t})

    st.markdown("""
    This app simulates the **Vasicek** short-rate model using Euler discretization:

    \[\mathrm{d}r_t = a (b - r_t)\,\mathrm{d}t + \sigma\,\mathrm{d}W_t\]

    It also overlays the **analytic mean** and a **±1σ band** for reference.
    """)

    # Chart: multiple paths with analytic mean band
    st.subheader("Simulated short-rate paths")
    base = alt.Chart(df_paths).transform_fold(
        fold=[c for c in df_paths.columns if c != "t"],
        as_=["series", "value"]
    ).mark_line(opacity=0.6).encode(
        x=alt.X("t:Q", title="Time (years)"),
        y=alt.Y("value:Q", title="Short rate"),
        detail="series:N"
    )

    band = alt.Chart(df_summary).mark_area(opacity=0.2).encode(
        x="t:Q",
        y="analytic_minus_1sd:Q",
        y2="analytic_plus_1sd:Q"
    )

    mean_line = alt.Chart(df_summary).mark_line(strokeWidth=2).encode(
        x="t:Q",
        y="analytic_mean:Q"
    )

    st.altair_chart(band + mean_line + base, use_container_width=True)

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
    st.download_button("Download simulated paths as CSV", csv, file_name="vasicek_paths.csv", mime="text/csv")

    # Save all artifacts as a zip on demand
    if st.button("Create repo ZIP (app, module, README, requirements)"):
        import zipfile, os, io, textwrap

        # Prepare files in-memory
        files = {}

        files["app.py"] = open("app.py", "r", encoding="utf-8").read()
        files["vasicek.py"] = open("vasicek.py", "r", encoding="utf-8").read()
        files["README.md"] = """# Vasicek Model Visualizer (Streamlit)

This project provides an interactive Streamlit app to **simulate and visualize** the Vasicek short-rate model.

## Model
The Vasicek SDE is:
\\[ \mathrm{d}r_t = a (b - r_t)\,\mathrm{d}t + \sigma\,\mathrm{d}W_t \\]

- **a**: speed of mean reversion
- **b**: long-term mean
- **\\sigma**: volatility
- **r\\_0**: initial short rate

The app uses **Euler discretization** to simulate paths and overlays the **analytic mean** and a **±1σ band**:

- Mean:  _E[r_t] = b + (r_0 - b) e^{-a t}_
- Variance:  _Var[r_t] = (\\sigma^2 / (2a)) (1 - e^{-2 a t})_  (for _a > 0_).

## Files
- `app.py` — Streamlit UI for parameter input, charts (Altair), and downloads.
- `vasicek.py` — Simulation utilities: single/multi-path simulators and analytic moments.
- `requirements.txt` — Python dependencies.
- `README.md` — This file.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes & Attribution
- The basic Vasicek Euler simulation structure is based on an example from Erik Rasin’s blog post “Python for Interest Rate Risk Management in ALM” (July 14, 2024). The implementation here expands it to multi-path simulation and adds analytic overlays and a Streamlit UI.
- Blog reference: https://www.erikrasin.io/blog/python-alm
"""

        files["requirements.txt"] = """streamlit
numpy
pandas
altair
"""

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, content in files.items():
                z.writestr(name, content)

        st.download_button(
            "Download repo ZIP",
            data=zip_buf.getvalue(),
            file_name="vasicek_streamlit_app.zip",
            mime="application/zip"
        )
