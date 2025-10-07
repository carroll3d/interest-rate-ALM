
# Short-Rate Model Visualizer (Streamlit) — Vasicek & CIR

Interactive Streamlit app to simulate and visualize two one-factor short-rate models:

- **Vasicek:**  d r_t = a (b - r_t) dt + σ dW_t  
- **CIR:**      d r_t = κ (θ - r_t) dt + σ √r_t dW_t

Uses Euler-type discretizations for simulation, overlays analytic means (and ±1σ bands where a closed-form variance is used).

## Files
- `app.py` — Streamlit UI with model dropdown, charts, and downloads.
- `vasicek.py` — Vasicek simulators and analytic moments.
- `cir.py` — CIR simulators and analytic moments.
- `requirements.txt` — Python dependencies.
- `README.md` — This file.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes & Attribution
- Based on examples from Erik Rasin’s blog post “Python for Interest Rate Risk Management in ALM”, adapted/extended here.
- Blog reference: https://www.erikrasin.io/blog/python-alm
