
#!/usr/bin/env python3
"""
Synthetic coffee roast curve generator.

This script creates plausible roast curves (BT/ET/RoR + control settings)
from high-level roast parameters such as charge temp, ambient temp, batch
mass, moisture content, and altitude.

Important:
- These curves are heuristic / illustrative, not physics-grade.
- They are meant as priors for experimentation and UI/mock-model work.
- "Good roast" presets included here are synthetic examples based on common
  roast-shape heuristics, not a claim that they are universally optimal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RoastParams:
    name: str = "custom"
    charge_temp_c: float = 200.0
    ambient_temp_c: float = 22.0
    batch_mass_g: float = 250.0
    moisture_pct: float = 10.5
    altitude_m: float = 1200.0
    target_drop_temp_c: float = 210.0
    dev_ratio: float = 0.18
    drying_share: float = 0.47
    roast_style: str = "balanced"  # light | balanced | developed
    time_step_s: int = 1
    noise: float = 0.0


PRESETS = {
    "good-light": RoastParams(
        name="good-light",
        charge_temp_c=198,
        ambient_temp_c=22,
        batch_mass_g=200,
        moisture_pct=10.7,
        altitude_m=1600,
        target_drop_temp_c=205,
        dev_ratio=0.14,
        drying_share=0.46,
        roast_style="light",
    ),
    "good-balanced": RoastParams(
        name="good-balanced",
        charge_temp_c=202,
        ambient_temp_c=22,
        batch_mass_g=250,
        moisture_pct=10.5,
        altitude_m=1400,
        target_drop_temp_c=210,
        dev_ratio=0.18,
        drying_share=0.47,
        roast_style="balanced",
    ),
    "good-developed": RoastParams(
        name="good-developed",
        charge_temp_c=205,
        ambient_temp_c=22,
        batch_mass_g=250,
        moisture_pct=10.3,
        altitude_m=1100,
        target_drop_temp_c=215,
        dev_ratio=0.22,
        drying_share=0.48,
        roast_style="developed",
    ),
}


STYLE_BIAS = {
    "light": dict(time=-30, fc_shift=-8, early_heat=1.03, fan_bias=0.98),
    "balanced": dict(time=0, fc_shift=0, early_heat=1.00, fan_bias=1.00),
    "developed": dict(time=40, fc_shift=5, early_heat=0.98, fan_bias=1.05),
}


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def phase_curve(t: np.ndarray, t0: float, t1: float, y0: float, y1: float) -> np.ndarray:
    x = (t - t0) / (t1 - t0)
    return y0 + (y1 - y0) * smoothstep(x)


def estimate_total_time_s(p: RoastParams) -> int:
    # Coarse heuristic: heavier batch + higher moisture + high altitude + lower charge = longer roast.
    total = 575.0
    total += 0.35 * (p.batch_mass_g - 250.0)
    total += 16.0 * (p.moisture_pct - 10.5)
    total += 0.012 * (p.altitude_m - 1200.0)
    total += -1.8 * (p.charge_temp_c - 200.0)
    total += 2.5 * (p.target_drop_temp_c - 210.0)
    total += STYLE_BIAS[p.roast_style]["time"]
    return int(np.clip(total, 420, 840))


def make_control_schedule(t: np.ndarray, dry_end_s: int, fc_s: int, drop_s: int, p: RoastParams):
    early_heat = 92 * STYLE_BIAS[p.roast_style]["early_heat"]
    mid_heat = 76 * STYLE_BIAS[p.roast_style]["early_heat"]
    late_heat = 55 * STYLE_BIAS[p.roast_style]["early_heat"]
    fan_bias = STYLE_BIAS[p.roast_style]["fan_bias"]

    heat = np.piecewise(
        t,
        [t < dry_end_s * 0.45,
         (t >= dry_end_s * 0.45) & (t < dry_end_s),
         (t >= dry_end_s) & (t < fc_s),
         (t >= fc_s) & (t < drop_s)],
        [
            lambda x: early_heat - 0.03 * x,
            lambda x: early_heat - 0.03 * dry_end_s * 0.45 - 0.04 * (x - dry_end_s * 0.45),
            lambda x: mid_heat - 0.025 * (x - dry_end_s),
            lambda x: late_heat - 0.01 * (x - fc_s),
        ],
    )
    heat = np.clip(heat, 30, 100)

    fan = np.piecewise(
        t,
        [t < dry_end_s * 0.7,
         (t >= dry_end_s * 0.7) & (t < fc_s),
         (t >= fc_s) & (t < drop_s)],
        [
            lambda x: 15 + 0.005 * x,
            lambda x: 30 + 0.01 * (x - dry_end_s * 0.7),
            lambda x: 48 + 0.02 * (x - fc_s),
        ],
    )
    fan = np.clip(fan * fan_bias, 10, 100)
    return heat, fan


def generate_curve(p: RoastParams) -> tuple[pd.DataFrame, dict]:
    total_s = estimate_total_time_s(p)
    t = np.arange(0, total_s + p.time_step_s, p.time_step_s)

    fc_temp_c = 196.0 + 0.002 * (p.altitude_m - 1200.0) + STYLE_BIAS[p.roast_style]["fc_shift"]
    fc_temp_c = float(np.clip(fc_temp_c, 191.5, 201.0))
    dry_end_temp_c = 160.0

    # Phase times
    drying_share = float(np.clip(p.drying_share, 0.40, 0.55))
    dev_ratio = float(np.clip(p.dev_ratio, 0.10, 0.28))

    dry_end_s = int(total_s * drying_share)
    dev_s = int(total_s * dev_ratio)
    fc_s = int(total_s - dev_s)
    if fc_s <= dry_end_s + 90:
        fc_s = dry_end_s + 90
    drop_s = total_s

    # Turning point: simplified bean temperature dip after charge
    tp_s = int(np.clip(75 + 0.1 * (p.batch_mass_g - 250) + 3.0 * (p.moisture_pct - 10.5), 55, 110))
    start_bt = p.ambient_temp_c + 15.0
    tp_temp = start_bt - np.clip(10 + 0.02 * p.batch_mass_g + 0.8 * (p.moisture_pct - 10.5), 8, 18)

    bt = np.zeros_like(t, dtype=float)
    # charge -> turning point dip
    bt[t <= tp_s] = phase_curve(t[t <= tp_s], 0, tp_s, start_bt, tp_temp)
    # turning point -> dry end
    mask = (t > tp_s) & (t <= dry_end_s)
    bt[mask] = phase_curve(t[mask], tp_s, dry_end_s, tp_temp, dry_end_temp_c)
    # dry end -> first crack
    mask = (t > dry_end_s) & (t <= fc_s)
    bt[mask] = phase_curve(t[mask], dry_end_s, fc_s, dry_end_temp_c, fc_temp_c)
    # first crack -> drop
    mask = t > fc_s
    bt[mask] = phase_curve(t[mask], fc_s, drop_s, fc_temp_c, p.target_drop_temp_c)

    # Add slight physically-plausible curvature via thermal drag
    bt += 1.8 * np.log1p(t / 60.0) - 1.2 * np.log1p(np.maximum(t - fc_s, 0) / 45.0)

    if p.noise > 0:
        rng = np.random.default_rng(42)
        bt += rng.normal(0, p.noise, size=len(bt))

    # Environmental temp (ET) stays above BT with shrinking offset late in roast
    et_offset = np.interp(t, [0, dry_end_s, fc_s, drop_s], [85, 60, 38, 28])
    et = bt + et_offset

    # Smooth RoR in C/min
    ror = np.gradient(bt, p.time_step_s) * 60.0
    # Gently taper RoR to avoid end spikes
    taper = np.interp(t, [0, dry_end_s, fc_s, drop_s], [1.05, 1.00, 0.92, 0.82])
    ror *= taper

    heat, fan = make_control_schedule(t, dry_end_s, fc_s, drop_s, p)

    # Approx "yellowing" marker as midpoint in drying phase
    yellow_s = int(tp_s + 0.65 * (dry_end_s - tp_s))

    events = np.full(len(t), "", dtype=object)
    event_map = {
        0: "charge",
        tp_s: "turning_point",
        yellow_s: "yellow",
        dry_end_s: "dry_end",
        fc_s: "first_crack_start",
        drop_s: "drop",
    }
    for sec, name in event_map.items():
        idx = np.argmin(np.abs(t - sec))
        events[idx] = name

    df = pd.DataFrame({
        "time_s": t,
        "time_mmss": [f"{int(x//60):02d}:{int(x%60):02d}" for x in t],
        "bean_temp_c": np.round(bt, 2),
        "env_temp_c": np.round(et, 2),
        "ror_c_per_min": np.round(ror, 2),
        "heat_pct": np.round(heat, 1),
        "fan_pct": np.round(fan, 1),
        "event": events,
    })

    meta = {
        "name": p.name,
        "dry_end_s": dry_end_s,
        "first_crack_s": fc_s,
        "drop_s": drop_s,
        "turning_point_s": tp_s,
        "yellow_s": yellow_s,
        "fc_temp_c": round(fc_temp_c, 1),
        "drop_temp_c": round(float(p.target_drop_temp_c), 1),
    }
    return df, meta


def plot_curve(df: pd.DataFrame, meta: dict, out_path: Path | None = None, title: str | None = None):
    t_min = df["time_s"].to_numpy() / 60.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_min, df["bean_temp_c"], label="BT (bean temp)")
    ax.plot(t_min, df["env_temp_c"], label="ET (environment temp)")
    ax2 = ax.twinx()
    ax2.plot(t_min, df["ror_c_per_min"], linestyle="--", label="RoR (C/min)")

    for key, label in [("dry_end_s", "Dry End"), ("first_crack_s", "1C"), ("drop_s", "Drop")]:
        x = meta[key] / 60.0
        ax.axvline(x=x, linestyle=":", linewidth=1)
        ax.text(x + 0.03, ax.get_ylim()[0] + 5, label, rotation=90, va="bottom")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (C)")
    ax2.set_ylabel("RoR (C/min)")
    ax.set_title(title or meta["name"])
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def save_profile(df: pd.DataFrame, meta: dict, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}.meta.txt"
    df.to_csv(csv_path, index=False)
    plot_curve(df, meta, png_path, title=stem)
    meta_lines = [f"{k}: {v}" for k, v in meta.items()]
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")


def generate_demo_set(out_dir: Path):
    summary_rows = []
    for preset_name in ["good-light", "good-balanced", "good-developed"]:
        p = PRESETS[preset_name]
        df, meta = generate_curve(p)
        save_profile(df, meta, out_dir, preset_name)
        summary_rows.append({
            "preset": preset_name,
            "dry_end": f'{meta["dry_end_s"]//60:02d}:{meta["dry_end_s"]%60:02d}',
            "first_crack": f'{meta["first_crack_s"]//60:02d}:{meta["first_crack_s"]%60:02d}',
            "drop": f'{meta["drop_s"]//60:02d}:{meta["drop_s"]%60:02d}',
            "drop_temp_c": meta["drop_temp_c"],
        })

    # Combined overview chart
    fig, ax = plt.subplots(figsize=(11, 7))
    for preset_name in ["good-light", "good-balanced", "good-developed"]:
        df = pd.read_csv(out_dir / f"{preset_name}.csv")
        ax.plot(df["time_s"] / 60.0, df["bean_temp_c"], label=preset_name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bean temperature (C)")
    ax.set_title("Synthetic illustrative 'good roast' BT curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "good_roast_examples.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_dir / "good_roast_examples_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic coffee roast curves.")
    parser.add_argument("--out-dir", type=Path, default=Path("roast_outputs"))
    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--name", default="custom")
    parser.add_argument("--charge-temp-c", type=float, default=200.0)
    parser.add_argument("--ambient-temp-c", type=float, default=22.0)
    parser.add_argument("--batch-mass-g", type=float, default=250.0)
    parser.add_argument("--moisture-pct", type=float, default=10.5)
    parser.add_argument("--altitude-m", type=float, default=1200.0)
    parser.add_argument("--target-drop-temp-c", type=float, default=210.0)
    parser.add_argument("--dev-ratio", type=float, default=0.18)
    parser.add_argument("--drying-share", type=float, default=0.47)
    parser.add_argument("--roast-style", choices=["light", "balanced", "developed"], default="balanced")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--demo-set", action="store_true", help="Generate illustrative preset curves.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_set:
        generate_demo_set(args.out_dir)

    if args.preset:
        p = PRESETS[args.preset]
    else:
        p = RoastParams(
            name=args.name,
            charge_temp_c=args.charge_temp_c,
            ambient_temp_c=args.ambient_temp_c,
            batch_mass_g=args.batch_mass_g,
            moisture_pct=args.moisture_pct,
            altitude_m=args.altitude_m,
            target_drop_temp_c=args.target_drop_temp_c,
            dev_ratio=args.dev_ratio,
            drying_share=args.drying_share,
            roast_style=args.roast_style,
            noise=args.noise,
        )

    df, meta = generate_curve(p)
    save_profile(df, meta, args.out_dir, p.name)


if __name__ == "__main__":
    main()
