"""Delt-IEt Turbofan Engine — núcleo para análise de trajetórias CMAPSS.

Implementação pública com parâmetros de referência (W=10, H=20, percentile=95).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = [f's{i}' for i in range(1, 22)]


def shannon_entropy(window: np.ndarray, bins: int = 10) -> float:
    """Entropia de Shannon da distribuição de amplitudes (janela deslizante)."""
    vals = np.array(window, dtype=float)
    if vals.std() < 1e-10:
        return 0.0
    counts, _ = np.histogram(vals, bins=bins)
    counts = counts.astype(float) + 1e-10
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def deltiet_single(series: np.ndarray, W: int = 10, H: int = 20) -> np.ndarray:
    """DeltIEt sobre uma série de sensor única.

    1. Entropia deslizante Sₜ com janela W.
    2. Segunda derivada discreta d²Sₜ.
    3. Z-score adaptativo Zₜ com janela H.
    4. DeltIEt = Zₜ · |d²Sₜ|
    """
    n = len(series)
    S = np.array([
        shannon_entropy(series[max(0, t - W):t]) if t >= W else 0.0
        for t in range(n)
    ])
    d2S = np.zeros(n)
    for t in range(2, n):
        d2S[t] = (S[t] - S[t - 1]) - (S[t - 1] - S[t - 2])
    Z = np.zeros(n)
    for t in range(H, n):
        w = S[t - H:t]
        s = w.std()
        if s > 1e-10:
            Z[t] = (S[t] - w.mean()) / s
    return Z * np.abs(d2S), S


def analyze_engine(
    df_unit: pd.DataFrame,
    W: int = 10,
    H: int = 20,
    percentile: int = 95,
    warmup: int = 20,
) -> dict | None:
    """Processa a trajetória de um único motor.

    Retorna dict com sinal combinado, threshold, eventos e métricas de alarme.
    """
    active = [c for c in SENSOR_COLS if c in df_unit.columns
              and float(df_unit[c].std()) > 0.01]
    if not active:
        return None

    n = len(df_unit)
    abs_signals = np.zeros((len(active), n))
    entropy_matrix = np.zeros((len(active), n))

    for i, col in enumerate(active):
        sig, ent = deltiet_single(df_unit[col].values, W=W, H=H)
        abs_signals[i] = np.abs(sig)
        entropy_matrix[i] = ent

    # Agregação multivariada: máximo entre sensores ativos
    combined = np.max(abs_signals, axis=0)

    theta = float(np.percentile(combined, percentile))

    events = np.zeros(n, dtype=int)
    if warmup < n:
        events[warmup:] = (combined[warmup:] > theta).astype(int)

    cycles = df_unit['cycle'].values
    first_alarm_idx = None
    for t in range(warmup, n):
        if events[t] == 1:
            first_alarm_idx = t
            break

    first_alarm_cycle = int(cycles[first_alarm_idx]) if first_alarm_idx is not None else None
    max_cycle = int(cycles[-1])
    rul_at_alarm = (max_cycle - first_alarm_cycle) if first_alarm_cycle is not None else None

    return {
        'cycles': cycles,
        'combined': combined,
        'theta': theta,
        'events': events,
        'first_alarm_cycle': first_alarm_cycle,
        'first_alarm_idx': first_alarm_idx,
        'max_cycle': max_cycle,
        'rul_at_alarm': rul_at_alarm,
        'active_sensors': len(active),
        'entropy_matrix': entropy_matrix,
        'active_sensor_names': active,
    }


def analyze_fleet(
    df: pd.DataFrame,
    W: int = 10,
    H: int = 20,
    percentile: int = 95,
    warmup: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Processa todos os motores do dataset.

    Retorna (fleet_df, per_engine_dict).
    """
    units = sorted(df['unit'].unique())
    rows = []
    per_engine: dict[int, dict] = {}

    for unit in units:
        df_unit = (df[df['unit'] == unit]
                   .sort_values('cycle')
                   .reset_index(drop=True))
        eng = analyze_engine(df_unit, W=W, H=H,
                             percentile=percentile, warmup=warmup)
        if eng is None:
            continue
        per_engine[int(unit)] = eng
        rows.append({
            'unit': int(unit),
            'max_cycle': eng['max_cycle'],
            'first_alarm_cycle': eng['first_alarm_cycle'],
            'rul_at_first_alarm': eng['rul_at_alarm'],
            'detected': 1 if eng['first_alarm_cycle'] is not None else 0,
            'active_sensors': eng['active_sensors'],
        })

    fleet_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return fleet_df, per_engine


def load_cmapss(file_obj) -> pd.DataFrame:
    """Carrega arquivo CMAPSS (.txt, espaço separado, sem cabeçalho)."""
    df = pd.read_csv(file_obj, sep=r'\s+', header=None)
    df = df.dropna(axis=1, how='all')
    n_cols = min(len(df.columns), len(COLS))
    df.columns = COLS[:n_cols]
    df['unit'] = df['unit'].astype(int)
    df['cycle'] = df['cycle'].astype(int)
    return df
