# python test_stockout_switch_unit_compare.py

import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpsim.sim import Sim, MultiSim
from fpsim.parameters import pars
from stockout_switch import StockoutSwitchIntervention


class TestStockoutSwitch(unittest.TestCase):
    def test_switching_logic_from_methods_1_and_3(self):
        # 1. Define switch matrix (use example ids; in real use, import the full map)
        switch_matrix = {
            7: [4, 9, 0],  # Implants -> Condoms, Other modern, None
            3: [4, 9, 0],  # Injectables -> Condoms, Other modern, None
        }

        # 2. Create stockout scenario for method 7 and 3 (100%)
        stockout_probs = {year: {7: 1.0, 3: 1.0} for year in range(2025, 2031)}
        stockout = StockoutSwitchIntervention(stockout_probs, switch_matrix, seed=42)

        # 3. Set up sims: with and without intervention
        p1 = pars(location="senegal", start_year=2020, end_year=2030, n_agents=1000)
        p2 = pars(location="senegal", start_year=2020, end_year=2030, n_agents=1000)

        sim_no_stockout = Sim(pars=p1, label="No stockout")
        sim_with_switch = Sim(pars=p2, label="With stockout switch")
        sim_with_switch['interventions'] = [stockout]

        # 4. Run both simulations
        msim = MultiSim(sims=[sim_no_stockout, sim_with_switch])
        msim.run()

        # 5. In 2025+, confirm no method 1 or 3 remain, and most users are still on contraception
        for ti in range(sim_with_switch.npts):
            year = sim_with_switch.ind2year(ti)
            if 2025 <= year <= 2030:
                sim_with_switch.ti = ti
                m = sim_with_switch.people.method
                self.assertEqual(
                    ((m == 1) | (m == 3)).sum(), 0,
                    f"Still found users on method 1 or 3 at timestep {ti} in {year}"
                )
                # Optional: check that many users are still using some method (not all dropped out)
                on_contra_rate = (m > 0).mean()
                self.assertGreater(
                    on_contra_rate, 0.6,
                    f"Too many users dropped out instead of switching at year {year} (only {on_contra_rate:.2%} on method)"
                )

        # 6. Plot CPR results
        msim.plot(to_plot='cpr', plot_sims=True)
        plt.suptitle("CPR comparison with and without stockout switch (method 1 and 3, 100%)", fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("You passed the unittest")
