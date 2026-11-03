import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def format_median(val, unit):
    if unit.lower() == "da":
        return f"{val:.6f}"
    return f"{val:.2f}"


def parse_filename(fname):
    m = re.match(r"(.+?)_(.+?)_([0-9.]+)_deviation_(before|after)_calib\.npy$", fname)
    if m is None:
        return None
    return {"group": m.group(1), "sample": m.group(2), "mass": float(m.group(3)), "state": m.group(4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare deviations before and after calibration.")
    parser.add_argument("input_dir", help="Directory containing deviation npy files")
    parser.add_argument("-result_dir", default=None)
    parser.add_argument("-unit", default="ppm")
    parser.add_argument("-plot", type=int, default=0)
    args = parser.parse_args()

    if args.result_dir is None:
        args.result_dir = os.path.join(args.input_dir, "boxplots")
    os.makedirs(args.result_dir, exist_ok=True)

    data = {}
    for fname in os.listdir(args.input_dir):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        key = (parsed["group"], parsed["sample"], parsed["mass"])
        if key not in data:
            data[key] = {}
        data[key][parsed["state"]] = os.path.join(args.input_dir, fname)

    masses = sorted(set(key[2] for key in data.keys()))
    print(f"Found {len(masses)} masses")

    for mass in masses:
        print(f"Processing mass {mass}")
        samples = []
        before_all = []
        after_all = []

        for (group, sample, m), files in sorted(data.items()):
            if m != mass:
                continue
            if "before" not in files or "after" not in files:
                print(f"Skipping {group}_{sample}_{mass}: missing before/after file")
                continue
            before = np.load(files["before"])
            after = np.load(files["after"])
            before = np.abs(before[np.isfinite(before)])
            after = np.abs(after[np.isfinite(after)])
            if len(before) == 0 or len(after) == 0:
                continue
            before_all.extend(before)
            after_all.extend(after)
            samples.append({"name": f"{group}_{sample}", "before": before, "after": after})

        if len(samples) == 0:
            print(f"No valid samples for mass {mass}")
            continue

        # Per-sample boxplot
        fig_width = max(8, len(samples) * 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        box_data = []
        positions = []
        tick_positions = []
        tick_labels = []

        for idx, sample in enumerate(samples):
            center = idx * 3
            before_pos = center - 0.4
            after_pos = center + 0.4
            positions.extend([before_pos, after_pos])
            box_data.extend([sample["before"], sample["after"]])
            tick_positions.append(center)
            before_med = np.median(sample["before"])
            after_med = np.median(sample["after"])
            tick_labels.append(f"{sample['name']}\n{format_median(before_med, args.unit)} → {format_median(after_med, args.unit)}")

        bp = ax.boxplot(box_data, positions=positions, widths=0.7, showfliers=False, patch_artist=True, medianprops=dict(color='black'))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("lightcoral" if i % 2 == 0 else "lightgreen")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel(f"Absolute deviation ({args.unit})")
        ax.set_title(f"Mass {mass}: per-sample calibration comparison")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color="lightcoral", label="Before calibration"),
                          plt.Rectangle((0, 0), 1, 1, color="lightgreen", label="After calibration")]
        ax.legend(handles=legend_handles)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, f"mass_{mass}_sample_comparison.svg"))
        if args.plot:
            plt.show()
        plt.close()

        # Combined boxplot
        before_all = np.asarray(before_all)
        after_all = np.asarray(after_all)
        median_before = np.median(before_all)
        median_after = np.median(after_all)
        plt.figure(figsize=(5, 6))
        bp = plt.boxplot([before_all, after_all],
                         labels=[f"Before\n({format_median(median_before, args.unit)})",
                                 f"After\n({format_median(median_after, args.unit)})"],
                         showfliers=False, patch_artist=True, medianprops=dict(color='black'))
        bp["boxes"][0].set_facecolor("lightcoral")
        bp["boxes"][1].set_facecolor("lightgreen")
        plt.ylabel(f"Absolute deviation ({args.unit})")
        plt.title(f"Mass {mass}\nMedian improvement = {format_median(median_before - median_after, args.unit)} {args.unit}")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, f"mass_{mass}_combined.svg"))
        if args.plot:
            plt.show()
        plt.close()

    print("Done.")