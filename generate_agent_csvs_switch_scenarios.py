# python generate_agent_csvs_switch_scenarios.py

import os
import pandas as pd
import numpy as np
import fpsim as fp
from stockout_switch import StockoutSwitchIntervention
from fpsim.methods import make_methods

# --- Settings ---
location = "senegal"
n_agents = 1000
start_year = 2020
end_year = 2030
stockout_years = range(2025, 2031)
output_dir = "./agents_by_year_scenarios"
os.makedirs(output_dir, exist_ok=True)

# --- Method name mapping
method_defs = make_methods()
inv_method_map = {v: k for k, v in method_defs.method_map.items()}

# --- Scenario configurations
scenarios = {
    "baseline": None,
    "stockout_m7": StockoutSwitchIntervention(
        stockout_probs={y: {7: 1.0} for y in stockout_years},
        switch_matrix={7: [4, 9, 0]},
        seed=42
    ),
    "stockout_m3": StockoutSwitchIntervention(
        stockout_probs={y: {3: 1.0} for y in stockout_years},
        switch_matrix={3: [4, 9, 0]},
        seed=42
    ),
    "stockout_both10": StockoutSwitchIntervention(
        stockout_probs={y: {3: 0.1, 7: 0.1} for y in stockout_years},
        switch_matrix={3: [4, 9, 0], 7: [4, 9, 0]},
        seed=42
    )
}

# --- Run scenarios
for label, intervention in scenarios.items():
    print(f"\n🚀 Running scenario: {label}")

    sim = fp.Sim(
        location=location,
        start_year=start_year,
        end_year=end_year,
        n_agents=n_agents,
        label=label
    )
    if intervention:
        sim['interventions'] = [intervention]
    sim.run()

    for ti in range(sim.npts):
        year = sim.ind2year(ti)
        sim.ti = ti
        ppl = sim.people

        df = pd.DataFrame({
            'uid': ppl.uid,
            'age': ppl.age,
            'alive': ppl.alive,
            'sex': ppl.sex,
            'method_fpsim': ppl.method_fpsim if hasattr(ppl, 'method_fpsim') else ppl.method,
            'method_final': ppl.method,
        })

        df = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]
        df['method_fpsim_name'] = df['method_fpsim'].map(inv_method_map)
        df['method_final_name'] = df['method_final'].map(inv_method_map)
        df['switch'] = df['method_fpsim'] != df['method_final']
        df['discontinue'] = df['method_final'] == 0
        df['year'] = year

        filename = os.path.join(output_dir, f"{label}_{year}_agents.csv")
        df.to_csv(filename, index=False)
        print(f"✅ Saved: {filename}")
