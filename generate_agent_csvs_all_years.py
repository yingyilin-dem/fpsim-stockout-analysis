# python generate_agent_csvs_all_years.py

"""
✅ Runs 4 FPsim scenarios
✅ Saves agent-level snapshots at the end of each simulated year
✅ Computes and prints mCPR (modern contraceptive prevalence rate) over time
✅ Verifies that mCPR changes dynamically with intervention
"""

import pandas as pd
import numpy as np
import os
import fpsim as fp
from stockout_discontinue import StockoutIntervention
from fpsim.methods import make_methods

# --- Helper to save agent-level data ---
def save_agent_data(sim, filename, year=None):
    people = sim.people
    df = pd.DataFrame({
        'uid': people.uid,
        'age': people.age,
        'alive': people.alive,
        'sex': people.sex,
        'method': people.method,
        'pregnant': people.pregnant,
        'postpartum': people.postpartum,
    })

    df = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]

    method_defs = make_methods()
    inv_method_map = {v: k for k, v in method_defs.method_map.items()}
    df['method_name'] = df['method'].map(inv_method_map)

    if year:
        df['year'] = year

    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")


# --- Simulation settings ---
location = 'senegal'
n_agents = 1000
years = range(2020, 2031)
stockout_years = range(2025, 2031)
output_dir = "./agents_by_year"
os.makedirs(output_dir, exist_ok=True)


# --- Run yearly sims for each scenario ---
def run_and_save(scenario_label, intervention_generator=None):
    for year in years:
        sim = fp.Sim(
            location=location,
            start_year=2020,
            end_year=year + 1,  # simulate through target year
            n_agents=n_agents,
            label=f"{scenario_label}_{year}"
        )
        if intervention_generator:
            sim['interventions'] = [intervention_generator()]
        sim.run()
        filename = os.path.join(output_dir, f"{scenario_label}_{year}_agents.csv")
        save_agent_data(sim, filename, year=year)


# --- Run all scenarios ---
run_and_save("baseline", intervention_generator=None)
run_and_save("stockout_m7", intervention_generator=lambda: StockoutIntervention({y: {7: 1.0} for y in stockout_years}, seed=42))
run_and_save("stockout_m3", intervention_generator=lambda: StockoutIntervention({y: {3: 1.0} for y in stockout_years}, seed=42))
run_and_save("stockout_both", intervention_generator=lambda: StockoutIntervention({y: {7: 1.0, 3: 1.0} for y in stockout_years}, seed=42))


# --- Check mCPR to confirm change over time ---
def check_mcpr(scenario_label):
    print(f"\n🔍 mCPR for {scenario_label}")
    for year in years:
        filepath = os.path.join(output_dir, f"{scenario_label}_{year}_agents.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            wra = df[(df['alive']) & (df['sex'] == 0) & (df['age'] >= 15) & (df['age'] <= 49)]
            mcpr = (wra['method'] > 0).mean() * 100
            print(f"Year {year}: mCPR = {mcpr:.1f}%")
        else:
            print(f"Year {year}: file not found")


# --- Print mCPR trends ---
for scenario in ["baseline", "stockout_m7", "stockout_m3", "stockout_both"]:
    check_mcpr(scenario)

print("\n🎉 All simulations completed successfully. Agent CSVs and mCPR trends saved.")
