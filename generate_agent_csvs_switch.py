# python generate_agent_csvs_switch.py

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
output_dir = "./agents_by_year_switch"
os.makedirs(output_dir, exist_ok=True)

# --- Stockout and fallback logic ---
stockout_years = range(2025, 2031)
stockout_probs = {y: {3: 1.0, 7: 1.0} for y in stockout_years}
switch_matrix = {
    3: [4, 9, 0],
    7: [4, 9, 0],
}

intervention = StockoutSwitchIntervention(stockout_probs, switch_matrix, seed=42)

# --- Run simulation ---
sim = fp.Sim(
    location=location,
    start_year=start_year,
    end_year=end_year,
    n_agents=n_agents,
    label="StockoutSwitch"
)
sim['interventions'] = [intervention]
sim.run()

# --- Load method names
method_defs = make_methods()
inv_method_map = {v: k for k, v in method_defs.method_map.items()}

# --- Save agent-level data by year ---
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

    # Focus on WRA
    df = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]

    # Optional: method names and switch flags
    df['method_fpsim_name'] = df['method_fpsim'].map(inv_method_map)
    df['method_final_name'] = df['method_final'].map(inv_method_map)
    df['switch'] = df['method_fpsim'] != df['method_final']
    df['discontinue'] = df['method_final'] == 0
    df['year'] = year

    # Save
    outpath = os.path.join(output_dir, f"stockout_switch_{year}_agents.csv")
    df.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
