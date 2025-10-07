
# Vasicek Model Visualizer (Streamlit)

This project provides an interactive Streamlit app to **simulate and visualize** the Vasicek short-rate model.

## Model
The Vasicek SDE is:
\[ \mathrm{d}r_t = a (b - r_t)\,\mathrm{d}t + \sigma\,\mathrm{d}W_t \]

- **a**: speed of mean reversion
- **b**: long-term mean
- **\sigma**: volatility
- **r_0**: initial short rate

The app uses **Euler discretization** to simulate paths and overlays the **analytic mean** and a **±1σ band**:

- Mean:  _E[r_t] = b + (r_0 - b) e^{-a t}_
- Variance:  _Var[r_t] = (\sigma^2 / (2a)) (1 - e^{-2 a t})_  (for _a > 0_).

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
