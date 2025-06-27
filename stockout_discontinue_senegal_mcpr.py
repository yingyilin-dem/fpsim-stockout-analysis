# run with: python stockout_discontinue_senegal_mcpr.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fpsim as fp
from fpsim.sim import Sim
from fpsim.methods import make_methods
from stockout_discontinue import StockoutIntervention

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
    print(f"Saved: {filename}")


# --- Simulation settings ---
location = 'senegal'
start_year = 2020
end_year = 2040
n_agents = 1000
stockout_years = range(2025, 2031)


# --- Baseline simulation ---
sim_base = Sim(location=location, start_year=start_year, end_year=end_year, n_agents=n_agents, label='baseline')
sim_base.run()
save_agent_data(sim_base, 'baseline_2030_agents.csv', year=2030)


# --- Method 1 stockout ---
stockout_m1_probs = {year: {7: 1.0} for year in stockout_years}
stockout_m1 = StockoutIntervention(stockout_probs=stockout_m1_probs, seed=42)

sim_m1 = Sim(location=location, start_year=start_year, end_year=end_year, n_agents=n_agents, label='stockout_m1')
sim_m1['interventions'] = [stockout_m1]
sim_m1.run()
save_agent_data(sim_m1, 'stockout_m1_2030_agents.csv', year=2030)


# --- Method 3 stockout ---
stockout_m3_probs = {year: {3: 1.0} for year in stockout_years}
stockout_m3 = StockoutIntervention(stockout_probs=stockout_m3_probs, seed=42)

sim_m3 = Sim(location=location, start_year=start_year, end_year=end_year, n_agents=n_agents, label='stockout_m3')
sim_m3['interventions'] = [stockout_m3]
sim_m3.run()
save_agent_data(sim_m3, 'stockout_m3_2030_agents.csv', year=2030)


# --- Method 1 & 3 stockout ---
stockout_both_probs = {year: {7: 1.0, 3: 1.0} for year in stockout_years}
stockout_both = StockoutIntervention(stockout_probs=stockout_both_probs, seed=42)

sim_both = Sim(location=location, start_year=start_year, end_year=end_year, n_agents=n_agents, label='stockout_both')
sim_both['interventions'] = [stockout_both]
sim_both.run()
save_agent_data(sim_both, 'stockout_both_2030_agents.csv', year=2030)


# --- Plot mCPR for all scenarios ---
plt.figure(figsize=(12, 6))
plt.plot(sim_base.results['t'], sim_base.results['cpr'] * 100, label='Baseline (Baseline stockout)', linewidth=2)
plt.plot(sim_m1.results['t'], sim_m1.results['cpr'] * 100, label='100% stockout: Implant', linestyle='--')
plt.plot(sim_m3.results['t'], sim_m3.results['cpr'] * 100, label='100% stockout: Injectable', linestyle=':')
plt.plot(sim_both.results['t'], sim_both.results['cpr'] * 100, label='100% stockout: Both Implany & Injectable', linestyle='-.')
plt.axvspan(2025, 2030, color='gray', alpha=0.2, label='Stockout period')
plt.xlabel('Year')
plt.ylabel('mCPR (%)')
plt.title('Senegal: mCPR Impact of Method-Specific Stockout Scenarios (Discontinue)')
plt.ylim(0, 50)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(ticks=plt.xticks()[0], labels=[f"{int(t)}" for t in plt.xticks()[0]])
plt.xlim(2020, 2040)
plt.show()
