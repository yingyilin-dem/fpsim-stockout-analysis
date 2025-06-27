import numpy as np
from fpsim.interventions import Intervention
from fpsim.utils import bt

class StockoutSwitchIntervention(Intervention):
    """
    Discontinue or switch method use if method-specific stockout occurs.
    For agents who discontinue, attempt to switch to a fallback method (shorter-acting).
    """

    def __init__(self, stockout_probs: dict[int, dict[int, float]], switch_matrix: dict[int, list[int]], seed: int | None = None):
        super().__init__()
        self.stockout_probs = stockout_probs  # {year: {method_id: probability}}
        self.switch_matrix = switch_matrix    # {method_id: [fallback_method_ids]}
        self.rng = np.random.default_rng(seed)

    def apply(self, sim):
        print(f"[StockoutSwitch] apply() called at year {sim.y:.2f}")

        year = int(sim.y)
        if year not in self.stockout_probs:
            return

        probs_for_year = self.stockout_probs[year]
        ppl = sim.people
        current_methods = ppl.method.copy()

        for i in range(len(ppl)):
            m = int(current_methods[i])
            if m == 0:
                continue  # Not on a method

            p_stock = probs_for_year.get(m, 0.0)
            if p_stock > 0.0 and bt(p_stock):
                print(f"  [Stockout] Agent {i} on method {m} → discontinued")

                # Start by discontinuing
                ppl.method[i] = 0
                ppl.on_contra[i] = False

                # Then try to switch
                fallback_list = self.switch_matrix.get(m, [])
                for alt_m in fallback_list:
                    if alt_m == 0:
                        break
                    p_alt_stock = probs_for_year.get(alt_m, 0.0)
                    if p_alt_stock > 0.0 and bt(p_alt_stock):
                        continue  # also stocked out
                    ppl.method[i] = alt_m
                    ppl.on_contra[i] = True
                    print(f"    [Switch] Agent {i} switched → method {alt_m}")
                    break
