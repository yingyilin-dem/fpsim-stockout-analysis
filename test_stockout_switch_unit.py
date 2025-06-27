
# Run with: python -m unittest test_stockout_switch_unit.py

import unittest
import numpy as np
from fpsim.sim import Sim
from fpsim.parameters import pars
from stockout_switch import StockoutSwitchIntervention

class TestStockoutSwitch(unittest.TestCase):
    def test_no_method_3_or_7_after_stockout(self):
        # Define 100% stockout for methods 3 and 7 from 2025–2030
        stockout_years = range(2025, 2031)
        stockout_probs = {y: {3: 1.0, 7: 1.0} for y in stockout_years}
        switch_matrix = {
            3: [4, 9, 0],
            7: [4, 9, 0],
        }

        intervention = StockoutSwitchIntervention(stockout_probs, switch_matrix, seed=42)

        # Set up a 2020–2030 simulation
        p = pars(location="senegal", start_year=2020, end_year=2030, n_agents=1000)
        sim = Sim(pars=p, label="TestSwitch")
        sim["interventions"] = [intervention]
        sim.run()

        # Check method use in each year from 2025 onward
        for ti in range(sim.npts):
            year = sim.ind2year(ti)
            if 2025 <= year <= 2030:
                sim.ti = ti
                m = sim.people.method
                violating = np.where((m == 3) | (m == 7))[0]
                self.assertEqual(len(violating), 0, f"{len(violating)} users still on method 3 or 7 in year {year}")

if __name__ == '__main__':
    unittest.main()
