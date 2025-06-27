# run with: python stockout_switch_fpsim_senegal_mcpr.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fpsim as fp
from fpsim.sim import Sim
from fpsim.methods import StandardChoice, make_methods
from stockout_switch_fpsim import StockoutSwitchFPsimIntervention

# ✅ Make sure 'method' is tracked over time
import fpsim.defaults as fpd
if 'method' not in fpd.longitude_keys:
    fpd.longitude_keys.append('method')

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

# --- Safe debug function using longitudinal history ---
def print_stocked_out_users(sim, label, years=range(2025, 2031)):
    print(f"\n[DEBUG] Checking for method 3/7 use during stockout years in: {label}")

    try:
        method_log = sim.people.longitude['method']
    except KeyError:
        print("  ❌ 'method' not found in longitude. Did you add it to fpd.longitude_keys before sim.run()?")
        return

    for y in years:
        ti_index = int((y - sim['start_year']) * sim['tiperyear'])

        if method_log.ndim != 2 or method_log.shape[1] <= ti_index:
            print(f"  ⚠️ 'method' history not available at year {y} (ti_index={ti_index})")
            continue

        methods = method_log[:, ti_index]
        count_3 = np.sum(methods == 3)
        count_7 = np.sum(methods == 7)
        print(f"  Year {y}: {count_3} agents on method 3, {count_7} on method 7")

# --- Simulation settings ---
location = 'senegal'
start_year = 2020
end_year = 2040
n_agents = 1000
stockout_years = range(2025, 2031)

def setup_sim(label, intervention=None):
    sim = Sim(location=location, start_year=start_year, end_year=end_year, n_agents=n_agents, label=label)
    sim.pars['method_choice'] = StandardChoice(location=location)
    if intervention:
        sim['interventions'] = [intervention]
    return sim

# --- Scenario: Baseline ---
sim_base = setup_sim('baseline')
sim_base.run()
save_agent_data(sim_base, 'baseline_2030_agents.csv', year=2030)
print_stocked_out_users(sim_base, "Baseline")

# --- Scenario: 100% stockout of implants (method 7) ---
stockout_m1 = StockoutSwitchFPsimIntervention(
    stockout_probs={year: {7: 1.0} for year in stockout_years},
    seed=42
)
sim_m1 = setup_sim('stockout_m1', stockout_m1)
sim_m1.run()
save_agent_data(sim_m1, 'stockout_m1_2030_agents.csv', year=2030)
print_stocked_out_users(sim_m1, "Implant Stockout")

# --- Scenario: 100% stockout of injectables (method 3) ---
stockout_m3 = StockoutSwitchFPsimIntervention(
    stockout_probs={year: {3: 1.0} for year in stockout_years},
    seed=42
)
sim_m3 = setup_sim('stockout_m3', stockout_m3)
sim_m3.run()
save_agent_data(sim_m3, 'stockout_m3_2030_agents.csv', year=2030)
print_stocked_out_users(sim_m3, "Injectable Stockout")

# --- Scenario: 100% stockout of both implants & injectables ---
stockout_both = StockoutSwitchFPsimIntervention(
    stockout_probs={year: {7: 1.0, 3: 1.0} for year in stockout_years},
    seed=42
)
sim_both = setup_sim('stockout_both', stockout_both)
sim_both.run()
save_agent_data(sim_both, 'stockout_both_2030_agents.csv', year=2030)
print_stocked_out_users(sim_both, "Both Methods Stockout")

# --- Plot mCPR for all scenarios ---
plt.figure(figsize=(12, 6))
plt.plot(sim_base.results['t'], sim_base.results['cpr'] * 100, label='Baseline (no stockout)', linewidth=2)
plt.plot(sim_m1.results['t'], sim_m1.results['cpr'] * 100, label='100% stockout: Implant', linestyle='--')
plt.plot(sim_m3.results['t'], sim_m3.results['cpr'] * 100, label='100% stockout: Injectable', linestyle=':')
plt.plot(sim_both.results['t'], sim_both.results['cpr'] * 100, label='100% stockout: Both Implant & Injectable', linestyle='-.')

plt.axvspan(2025, 2030, color='gray', alpha=0.2, label='Stockout period')
plt.xlabel('Year')
plt.ylabel('mCPR (%)')
plt.title('Senegal: mCPR Impact of Method-Specific Stockout with FPsim Switching Logic')
plt.ylim(0, 50)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(ticks=plt.xticks()[0], labels=[f"{int(t)}" for t in plt.xticks()[0]])
plt.xlim(2020, 2040)
plt.show()
