# python -m tests.run_simulation

from fpsim.sim import Sim
from fpsim.parameters import pars
from fpsim.stockout import StockoutIntervention
from fpsim.analyzers import cpr_by_age

# 1. Define simulation parameters
p = pars(location="kenya", start_year=2020, end_year=2030, n_agents=1000)

# 2. Define stockout schedule
# Use method IDs (not labels); 1 might be Pill, 2 = IUD, etc.
stockout_probs = {
    2025: {1: 0.10, 2: 0.05},  # 10% chance pill stockout, 5% for IUD
    2026: {1: 0.15, 2: 0.08},
    2027: {1: 0.20, 2: 0.10},
}

# 3. Create the stockout intervention
stockout_int = StockoutIntervention(
    stockout_probs=stockout_probs,
    switch_if_stockout=False,  # Change to True to simulate method switching
    seed=42
)

# 4. Create an analyzer to track mCPR over time
analyzer = cpr_by_age()

# 5. Create and run the simulation
sim = Sim(
    pars=p,
    interventions=[stockout_int],
    analyzers=[analyzer]
)
sim.run()

# 6. Print mCPR over time
print("mCPR over time (total):")
for ti, cpr_val in enumerate(analyzer.total):
    year = sim.ind2year(ti)
    print(f"Year {year}: mCPR = {cpr_val:.3f}")
