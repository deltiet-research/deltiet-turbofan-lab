# Delt-IEt Turbofan Lab ✈️

Interactive environment for jet-engine degradation analysis using the **Delt-IEt** index.

Upload any NASA CMAPSS training file (`train_FD001.txt` … `train_FD004.txt`) and the tool computes a multivariate Delt-IEt signal for every engine trajectory — no failure labels, no training data, no engine-specific tuning.

## Results on full CMAPSS (FD001–FD004)

| Engines | Detected | Detection rate | Median RUL at alarm | Earliest alarm |
|---------|----------|----------------|---------------------|----------------|
| 709 | 709 | 100% | 165 cycles | 502 cycles |

Parameters used: W = 10, H = 20, Θ = 95th percentile, warmup = 20.

## Formula

```
DeltIEt(t) = Zₜ · |Δ²Sₜ|
```

Where:
- `Sₜ` = Shannon entropy of sensor values in a sliding window of size W
- `Δ²Sₜ` = second discrete derivative of Sₜ
- `Zₜ` = adaptive z-score of Sₜ over a lookback window of size H
- Multivariate aggregation: max across active sensors (std > 0.01)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Related

- 📓 Reference notebook: https://www.kaggle.com/code/ubirajaravasconcelos/delt-iet-what-is-happening-at-cycle-8
- 🔋 Battery version: https://deltiet-battery-lab.streamlit.app
- 📄 Research record: https://zenodo.org/records/20326175
- 🔒 Patent: BR 10 2026 012158 4 (pending)

## Limitation

The threshold Θ is computed retrospectively over each complete trajectory.
It does not represent an online or real-time detection system.

---

*Delt-IEt — independent research · Feira de Santana, Brasil*
