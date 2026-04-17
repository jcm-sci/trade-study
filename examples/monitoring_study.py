"""Monitoring station design study.

A multi-objective trade study over instrument settings for a remote
monitoring station.  The goal is to recover a known environmental
signal as faithfully as possible while keeping station cost low.

The observation model applies realistic signal-chain transformations
(anti-alias filtering, decimation, quantization, sensor noise) using
only numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_study import (
    Direction,
    Factor,
    FactorType,
    Observable,
    build_grid,
    extract_front,
    plot_front,
    plot_parallel,
    plot_scores,
    run_grid,
)

ASSET_DIR = "docs/assets"

# ── Ground-truth signal ────────────────────────────────────────────

# --8<-- [start:signal]
DURATION = 86400.0  # 24 hours [s]
DT_TRUE = 1.0  # true signal resolution [s]
N_TRUE = int(DURATION / DT_TRUE)
T_TRUE = np.arange(N_TRUE) * DT_TRUE  # time axis [s]

RNG = np.random.default_rng(42)


def generate_reference_signal() -> np.ndarray:
    """Build a synthetic 24-hour environmental signal.

    Components:
      - Diurnal cycle (period = 24 h)
      - Short-period oscillations (periods ≈ 5 min, 20 min)
      - Correlated noise (AR(1), φ = 0.995)

    Returns:
        1-D array of length N_TRUE.
    """
    t_hours = T_TRUE / 3600.0

    # Diurnal
    diurnal = 5.0 * np.sin(2.0 * np.pi * t_hours / 24.0)

    # Short-period oscillations (5 min and 20 min)
    t_min = T_TRUE / 60.0
    short = 1.2 * np.sin(2.0 * np.pi * t_min / 5.0) + 0.8 * np.cos(
        2.0 * np.pi * t_min / 20.0
    )

    # Correlated noise  (AR(1), φ = 0.995 → decorrelation ~ 200 s)
    phi = 0.995
    noise = np.empty(N_TRUE)
    noise[0] = RNG.normal()
    for i in range(1, N_TRUE):
        noise[i] = phi * noise[i - 1] + RNG.normal(scale=np.sqrt(1 - phi**2))

    return diurnal + short + noise


REFERENCE = generate_reference_signal()
# --8<-- [end:signal]


# ── Observation model ──────────────────────────────────────────────

# --8<-- [start:observation]
SENSOR_NOISE: dict[str, float] = {
    "field": 1.0,
    "lab": 0.1,
    "reference": 0.01,
}


def observe(
    signal: np.ndarray,
    sample_interval: int,
    adc_bits: int,
    sensor_grade: str,
    n_sensors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the instrument signal chain to a reference signal.

    Steps:
      1. Anti-alias low-pass filter (brick-wall in frequency domain).
      2. Decimation to the configured sample interval.
      3. Quantization to *adc_bits* resolution.
      4. Additive sensor noise, reduced by √n_sensors averaging.

    Args:
        signal: High-resolution reference signal.
        sample_interval: Seconds between samples (decimation factor).
        adc_bits: ADC bit depth.
        sensor_grade: One of "field", "lab", "reference".
        n_sensors: Number of co-located sensors to average.
        rng: Numpy random generator for reproducibility.

    Returns:
        Degraded observation array at the decimated sample rate.
    """
    # 1. Anti-alias filter (brick-wall at Nyquist of decimated rate)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=DT_TRUE)
    nyquist = 0.5 / sample_interval
    spectrum[freqs > nyquist] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(signal))

    # 2. Decimate
    decimated = filtered[::sample_interval]

    # 3. Quantize
    sig_range = float(np.ptp(signal)) * 1.1  # 10 % headroom
    n_levels = 2**adc_bits
    step = sig_range / n_levels
    mid = float(np.mean(signal))
    quantized = np.round((decimated - mid) / step) * step + mid

    # 4. Sensor noise (averaged over n_sensors)
    sigma = SENSOR_NOISE[sensor_grade] / np.sqrt(n_sensors)
    return quantized + rng.normal(scale=sigma, size=len(quantized))


# --8<-- [end:observation]


# ── Simulator and scorer ───────────────────────────────────────────


# --8<-- [start:world]
class StationSimulator:
    """Simulator that applies the instrument signal chain.

    Returns the full-resolution reference signal as 'truth' and
    the degraded, decimated observations as 'observations'.
    """

    def generate(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Generate truth and degraded observations.

        Args:
            config: Must contain sample_interval, adc_bits,
                sensor_grade, and n_sensors.

        Returns:
            Tuple of (reference_signal, degraded_observations).
        """
        seed = hash(frozenset(config.items())) % 2**32
        rng = np.random.default_rng(seed)
        obs = observe(
            REFERENCE,
            sample_interval=config["sample_interval"],
            adc_bits=config["adc_bits"],
            sensor_grade=config["sensor_grade"],
            n_sensors=config["n_sensors"],
            rng=rng,
        )
        return REFERENCE, {"observations": obs, "interval": config["sample_interval"]}


def _spectral_fidelity(truth: np.ndarray, reconstructed: np.ndarray) -> float:
    """Fraction of spectral power recovered in the 2.5-30 min band.

    Args:
        truth: Full-resolution reference signal.
        reconstructed: Signal interpolated back to the reference grid.

    Returns:
        Power ratio clamped to [0, 1].
    """
    ref_psd = np.abs(np.fft.rfft(truth)) ** 2
    rec_psd = np.abs(np.fft.rfft(reconstructed)) ** 2
    freqs = np.fft.rfftfreq(len(truth), d=DT_TRUE)
    band = (freqs >= 1 / 1800) & (freqs <= 1 / 150)  # 2.5 min - 30 min
    ref_power = float(np.sum(ref_psd[band]))
    rec_power = float(np.sum(rec_psd[band]))
    return min(rec_power / ref_power, 1.0) if ref_power > 0 else 0.0


def _station_cost(config: dict[str, Any]) -> float:
    """Additive cost model over instrument settings.

    Args:
        config: Instrument configuration dict.

    Returns:
        Total station cost in arbitrary units.
    """
    rate_cost = {1: 50, 5: 30, 15: 15, 60: 5, 300: 1}
    bits_cost = {8: 1, 12: 5, 16: 20, 24: 80}
    grade_cost = {"field": 10, "lab": 50, "reference": 200}
    return float(
        rate_cost[config["sample_interval"]]
        + bits_cost[config["adc_bits"]]
        + grade_cost[config["sensor_grade"]] * config["n_sensors"]
    )


class StationScorer:
    """Score observation quality and station cost."""

    def score(
        self,
        truth: Any,
        observations: Any,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Compute RMSE, spectral fidelity, and station cost.

        Args:
            truth: Full-resolution reference signal.
            observations: Dict with degraded signal and sample interval.
            config: Instrument configuration.

        Returns:
            Scores for rmse, spectral_fidelity, and station_cost.
        """
        obs = observations["observations"]
        interval = observations["interval"]

        # Interpolate observations back to the reference grid
        obs_times = np.arange(len(obs)) * interval
        reconstructed = np.interp(T_TRUE, obs_times, obs)

        # RMSE
        rmse = float(np.sqrt(np.mean((truth - reconstructed) ** 2)))

        # Spectral fidelity: power ratio in the short-period band
        fidelity = _spectral_fidelity(truth, reconstructed)

        # Station cost (arbitrary units)
        cost = _station_cost(config)

        return {
            "rmse": rmse,
            "spectral_fidelity": fidelity,
            "station_cost": float(cost),
        }


# --8<-- [end:world]

# ── Study definition ───────────────────────────────────────────────

# --8<-- [start:observables]
observables = [
    Observable("rmse", Direction.MINIMIZE),
    Observable("spectral_fidelity", Direction.MAXIMIZE),
    Observable("station_cost", Direction.MINIMIZE),
]
# --8<-- [end:observables]

# --8<-- [start:factors]
factors = [
    Factor("sample_interval", FactorType.DISCRETE, levels=[1, 5, 15, 60, 300]),
    Factor("adc_bits", FactorType.DISCRETE, levels=[8, 12, 16, 24]),
    Factor("sensor_grade", FactorType.DISCRETE, levels=["field", "lab", "reference"]),
    Factor("n_sensors", FactorType.DISCRETE, levels=[1, 2, 4, 8]),
]
# --8<-- [end:factors]


def _plot_reference_signal(plt: Any) -> None:
    """Save a 1-hour window of the ground-truth reference signal."""
    window = slice(3600, 7200)
    t_hours = T_TRUE[window] / 3600.0

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_hours, REFERENCE[window], linewidth=0.6, color="0.3")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Signal amplitude")
    ax.set_title("Reference signal (1-hour window)")
    fig.tight_layout()
    fig.savefig(f"{ASSET_DIR}/monitoring_signal.png", dpi=150, bbox_inches="tight")
    print("\nSaved monitoring_signal.png")
    plt.close(fig)


def _plot_best_vs_cheapest(
    plt: Any,
    best_cfg: dict[str, Any],
    cheap_cfg: dict[str, Any],
) -> None:
    """Compare best-RMSE and cheapest Pareto designs against truth."""
    seed_best = hash(frozenset(best_cfg.items())) % 2**32
    obs_best = observe(
        REFERENCE,
        sample_interval=best_cfg["sample_interval"],
        adc_bits=best_cfg["adc_bits"],
        sensor_grade=best_cfg["sensor_grade"],
        n_sensors=best_cfg["n_sensors"],
        rng=np.random.default_rng(seed_best),
    )
    seed_cheap = hash(frozenset(cheap_cfg.items())) % 2**32
    obs_cheap = observe(
        REFERENCE,
        sample_interval=cheap_cfg["sample_interval"],
        adc_bits=cheap_cfg["adc_bits"],
        sensor_grade=cheap_cfg["sensor_grade"],
        n_sensors=cheap_cfg["n_sensors"],
        rng=np.random.default_rng(seed_cheap),
    )

    fig, (ax_b, ax_c) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t_best = np.arange(len(obs_best)) * best_cfg["sample_interval"] / 3600.0
    t_cheap = np.arange(len(obs_cheap)) * cheap_cfg["sample_interval"] / 3600.0

    mask_ref = (T_TRUE / 3600.0 >= 1.0) & (T_TRUE / 3600.0 <= 2.0)
    mask_best = (t_best >= 1.0) & (t_best <= 2.0)
    mask_cheap = (t_cheap >= 1.0) & (t_cheap <= 2.0)

    ax_b.plot(
        T_TRUE[mask_ref] / 3600.0,
        REFERENCE[mask_ref],
        linewidth=0.5,
        color="0.7",
        label="Truth",
    )
    ax_b.plot(t_best[mask_best], obs_best[mask_best], linewidth=0.8, label="Best RMSE")
    ax_b.set_ylabel("Amplitude")
    ax_b.set_title(
        f"Best RMSE: interval={best_cfg['sample_interval']}s, "
        f"bits={best_cfg['adc_bits']}, "
        f"grade={best_cfg['sensor_grade']}, n={best_cfg['n_sensors']}",
    )
    ax_b.legend(loc="upper right")

    ax_c.plot(
        T_TRUE[mask_ref] / 3600.0,
        REFERENCE[mask_ref],
        linewidth=0.5,
        color="0.7",
        label="Truth",
    )
    ax_c.plot(
        t_cheap[mask_cheap],
        obs_cheap[mask_cheap],
        linewidth=0.8,
        color="C1",
        label="Cheapest",
    )
    ax_c.set_xlabel("Time [hours]")
    ax_c.set_ylabel("Amplitude")
    ax_c.set_title(
        f"Cheapest: interval={cheap_cfg['sample_interval']}s, "
        f"bits={cheap_cfg['adc_bits']}, "
        f"grade={cheap_cfg['sensor_grade']}, n={cheap_cfg['n_sensors']}",
    )
    ax_c.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(
        f"{ASSET_DIR}/monitoring_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved monitoring_comparison.png")
    plt.close(fig)


def main() -> None:
    """Run the monitoring station trade study and print results."""
    # --8<-- [start:run]
    grid = build_grid(factors, method="full")
    print(f"Full factorial grid: {len(grid)} configurations")

    results = run_grid(
        world=StationSimulator(),
        scorer=StationScorer(),
        grid=grid,
        observables=observables,
    )
    # --8<-- [end:run]

    # --8<-- [start:results]
    # Pareto front
    directions = [o.direction for o in observables]
    front_idx = extract_front(
        results.scores,
        directions,
    )
    print(f"Pareto front: {len(front_idx)} / {len(grid)} designs\n")

    print(
        f"{'Interval':>8s}  {'Bits':>4s}  {'Grade':>9s}  {'#Sens':>5s}  "
        f"{'RMSE':>6s}  {'Fidelity':>8s}  {'Cost':>6s}"
    )
    print("-" * 60)
    for i in front_idx:
        cfg = results.configs[i]
        rmse_val, fid, cost = results.scores[i]
        print(
            f"{cfg['sample_interval']:8d}  {cfg['adc_bits']:4d}  "
            f"{cfg['sensor_grade']:>9s}  {cfg['n_sensors']:5d}  "
            f"{rmse_val:6.3f}  {fid:8.4f}  {cost:6.0f}"
        )

    # Cheapest design on the front
    cheapest = front_idx[np.argmin(results.scores[front_idx, 2])]
    print(f"\nCheapest Pareto design: {results.configs[cheapest]}")
    print(
        f"  RMSE={results.scores[cheapest, 0]:.4f}  "
        f"fidelity={results.scores[cheapest, 1]:.4f}  "
        f"cost={results.scores[cheapest, 2]:.0f}"
    )

    # Best RMSE on the front
    best = front_idx[np.argmin(results.scores[front_idx, 0])]
    print(f"\nLowest-RMSE Pareto design: {results.configs[best]}")
    print(
        f"  RMSE={results.scores[best, 0]:.4f}  "
        f"fidelity={results.scores[best, 1]:.4f}  "
        f"cost={results.scores[best, 2]:.0f}"
    )
    # --8<-- [end:results]

    # --8<-- [start:plots]
    import matplotlib.pyplot as plt

    # ── Domain-specific: reference signal (1-hour window) ──────────
    _plot_reference_signal(plt)

    # ── Domain-specific: best vs cheapest Pareto designs ───────────
    _plot_best_vs_cheapest(plt, results.configs[best], results.configs[cheapest])

    # ── Trade-study plots ──────────────────────────────────────────
    # Pareto front scatter (3 objectives → pairwise matrix)
    fig_front, _ = plot_front(results, directions)
    fig_front.savefig(
        f"{ASSET_DIR}/monitoring_front.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved monitoring_front.png")
    plt.close(fig_front)

    # Parallel coordinates
    fig_par, _ = plot_parallel(results, directions)
    fig_par.savefig(
        f"{ASSET_DIR}/monitoring_parallel.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved monitoring_parallel.png")
    plt.close(fig_par)

    # RMSE strip plot
    fig_rmse, _ = plot_scores(results, "rmse", directions)
    fig_rmse.savefig(
        f"{ASSET_DIR}/monitoring_rmse.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved monitoring_rmse.png")
    plt.close(fig_rmse)
    # --8<-- [end:plots]


if __name__ == "__main__":
    main()
