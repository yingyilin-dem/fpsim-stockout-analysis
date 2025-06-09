
"""
- Applies a 100% stockout for method 1 and 3 in 2025
- So in 2025, no one should be using method 1 or method 3, out of the 100 simulations
- no one should be using method 1 or 3  
- one simulation set to 100 agents only (to reduce computation time)
- ATTENTION!!: why out of the 100 simulations, final pop size is all 100?? sth is not right...
- Find out fpsim tutorial: how do ppl specify number of runs???
"""

# python -m tests.test_stockout_multisim

# tests/test_stockout_multisim.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sciris import dcp
from fpsim.sim import Sim, MultiSim
from fpsim.parameters import pars
from fpsim.stockout import StockoutIntervention

def test_multisim_discontinue_methods_1_and_3_in_2025():
    p = pars(location="kenya", start_year=2020, end_year=2030, n_agents=100)
    stockout_probs = {2025: {1: 1.0, 3: 1.0}}
    stockout = StockoutIntervention(stockout_probs, seed=42)

    base = Sim(pars=p)
    base['interventions'] = [stockout]

    # ❗ FIX: manually create a list of 100 cloned sims
    sims = [dcp(base) for _ in range(100)]
    msim = MultiSim(sims=sims)
    msim.run()

    for i, sim in enumerate(msim.sims):
        for ti in range(sim.npts):
            year = sim.ind2year(ti)
            if year == 2025:
                sim.ti = ti
                m = sim.people.method
                count = ((m == 1) | (m == 3)).sum()
                assert count == 0, f"Run {i}: {count} people still using method 1 or 3 in 2025."

    print("Passed MultiSim: no one uses method 1 or 3 during 2025 in any of 100 simulations.")

if __name__ == '__main__':
    test_multisim_discontinue_methods_1_and_3_in_2025()
