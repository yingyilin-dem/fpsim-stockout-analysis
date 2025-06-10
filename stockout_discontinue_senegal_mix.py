# python stockout_discontinue_senegal_mix.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib import colormaps

# --- Config ---
scenarios = {
    "Baseline": "baseline",
    "Scenario 1: Implant 100% stockout": "stockout_m7",
    "Scenario 2: Injectable 100% stockout": "stockout_m3",
    "Scenario 3: Implant & Injectable 100% stockout": "stockout_both"
}
years = list(range(2020, 2031))
stockout_start, stockout_end = 2025, 2030
data_dir = "./agents_by_year"

# --- Load method mix from agent-level CSVs ---
def compute_full_method_mix(scenario_prefix):
    mix_by_year = {}
    for year in years:
        file = os.path.join(data_dir, f"{scenario_prefix}_{year}_agents.csv")
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file)
        df = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]
        df['method_name'] = df['method_name'].fillna("None")
        df.loc[df['method'] == 0, 'method_name'] = "None"
        mix = df['method_name'].value_counts(normalize=True).sort_index() * 100
        mix_by_year[year] = mix
    return mix_by_year

# --- Collect all method names ---
all_method_names = set()
scenario_mixes = {}
for label, prefix in scenarios.items():
    mix = compute_full_method_mix(prefix)
    scenario_mixes[label] = mix
    for m in mix.values():
        all_method_names.update(m.index)

# Sort with "None" last
all_method_names = sorted(all_method_names, key=lambda x: (x == "None", x))

# --- Assign default tab20 colors + custom overrides ---
cmap = colormaps.get_cmap("tab20")
colors = [cmap(i / max(1, len(all_method_names)-1)) for i in range(len(all_method_names)-1)] + ['#ffffff']
method_colors = dict(zip(all_method_names, colors))

# --- Manual overrides ---
for name in method_colors:
    if name.lower() == "pill":
        method_colors[name] = "#1f77b4"  # blue
    elif "sterilization" in name.lower() or "btl" in name.lower():
        method_colors[name] = "#ffd700"  # yellow

# --- Build legend handles (excluding 'None') ---
legend_handles = [Patch(facecolor=method_colors[m], label=m)
                  for m in all_method_names if m.lower() != "none"]
legend_handles.append(Patch(facecolor='gray', alpha=0.2, label='Stockout Period'))

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True, constrained_layout=True)
axes = axes.flatten()

for ax, (label, mix_dict) in zip(axes, scenario_mixes.items()):
    matrix = np.zeros((len(all_method_names), len(years)))
    for t_idx, year in enumerate(years):
        year_mix = mix_dict.get(year, pd.Series())
        for m_idx, method in enumerate(all_method_names):
            matrix[m_idx, t_idx] = year_mix.get(method, 0)
    color_list = [method_colors[m] for m in all_method_names]
    ax.stackplot(years, matrix, labels=all_method_names, colors=color_list, alpha=0.95)
    ax.set_title(label)
    ax.set_xlim(min(years), max(years))
    ax.set_ylim(0, 40)
    ax.set_ylabel("Share of All WRA (%)")
    ax.axvspan(stockout_start, stockout_end + 1, color='gray', alpha=0.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

axes[2].set_xlabel("Year")
axes[3].set_xlabel("Year")

# --- Layout & Legend ---
fig.subplots_adjust(right=0.8)
fig.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(0.82, 0.5),
    bbox_transform=fig.transFigure,
    title='Method'
)
fig.suptitle("Senegal: Method Mix (2020–2030) as Share of All WRA", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.8, 0.95])
plt.show()
