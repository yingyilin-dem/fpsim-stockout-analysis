"""
- Applies a 100% stockout for method 1 and 3 in 2025
- In 2025, no one should be using method 1 or method 3 (test discontinue)
- Also plot CPR results with and without the stockout
"""

# python -m test_stockout_unit_compare

import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpsim.sim import Sim, MultiSim
from fpsim.parameters import pars
from stockout import StockoutIntervention
import matplotlib.pyplot as plt

class TestStockoutDiscontinuation(unittest.TestCase):
    def test_methods_1_and_3_discontinued_in_2025(self):
        # 1. Simulation setup: 2020–2030, 1000 agents
        p1 = pars(location="kenya", start_year=2020, end_year=2030, n_agents=1000)
        p2 = pars(location="kenya", start_year=2020, end_year=2030, n_agents=1000)

        # 2. Sim WITHOUT stockout
        sim_no_stockout = Sim(pars=p1, label="No stockout")

        # 3. Sim WITH 100% stockout for methods 1 and 3 in 2025
        stockout_probs = {year: {1: 1.0, 3: 1.0} for year in range(2025, 2031)}
        stockout = StockoutIntervention(stockout_probs, seed=42)
        sim_stockout = Sim(pars=p2, label="With stockout")
        sim_stockout['interventions'] = [stockout]

        # 4. Run both in MultiSim
        msim = MultiSim(sims=[sim_no_stockout, sim_stockout])
        msim.run()

        # 5. Check: no one should be using method 1 or 3 in 2025 in stockout sim
        for ti in range(sim_stockout.npts):
            year = sim_stockout.ind2year(ti)
            if 2025 <= year <= 2030:
                sim_stockout.ti = ti
                m = sim_stockout.people.method
                count = ((m == 1) | (m == 3)).sum()
                self.assertEqual(
                    count, 0,
                    f"{count} people still on method 1 or 3 at timestep {ti} in 2025."
                )

        # 6. Plot: CPR/MCPR/ACPR, showing both simulations
        msim.plot(to_plot='cpr', plot_sims=True)
        plt.suptitle("CPR comparison with and without stockout (method 1 and 3, 100%)", fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("You passed the unittest")
