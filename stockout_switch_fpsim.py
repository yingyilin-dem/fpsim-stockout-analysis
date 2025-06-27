import numpy as np
from fpsim.interventions import Intervention
from fpsim.utils import bt  # Binomial trial

"""
- No re-selecting the just-discontinued method
- No selecting any stocked-out method
- One chance to switch only
- Otherwise → fallback to method = 0 (nonuser)
"""

class StockoutSwitchFPsimIntervention(Intervention):
    """
    If current method is stocked out, try to switch using FPsim logic.
    Stocked-out methods are excluded from the switching pool.
    Resample once only. If new method is also stocked out or same as discontinued → become non-user.
    """

    def __init__(self, stockout_probs: dict[int, dict[int, float]], seed: int | None = None):
        super().__init__()
        self.stockout_probs = stockout_probs  # {year: {method_idx: prob}}
        self.rng = np.random.default_rng(seed)

    def apply(self, sim):
        year = int(sim.y)
        if year not in self.stockout_probs:
            return

        ppl = sim.people
        probs_for_year = self.stockout_probs[year]
        method_choice = sim.pars['method_choice']

        # Map method index to weight index
        method_idx_to_weight_idx = {m.idx: i for i, m in enumerate(method_choice.methods.values())}

        for i in range(len(ppl)):
            m = int(ppl.method[i])
            if m == 0:
                continue  # Not using a method

            p_stock = probs_for_year.get(m, 0.0)
            if p_stock > 0.0 and bt(p_stock):
                print(f"[{year}] Agent {i} discontinued method {m}")
                ppl.method[i] = 0
                ppl.on_contra[i] = False

                # Modify weights to exclude stocked-out methods
                original_weights = method_choice.pars['method_weights'].copy()
                new_weights = original_weights.copy()
                for method_id, p in probs_for_year.items():
                    if p > 0.0 and method_id in method_idx_to_weight_idx:
                        new_weights[method_idx_to_weight_idx[method_id]] = 0.0
                method_choice.pars['method_weights'] = new_weights

                # Choose new method (resample once)
                mask = np.zeros(len(ppl), dtype=bool)
                mask[i] = True
                person = ppl.filter(mask)
                new_method = method_choice.choose_method(person)[0]

                # Restore original weights
                method_choice.pars['method_weights'] = original_weights

                # Check for invalid switch
                if new_method == m:
                    print(f"    Switched back to method {new_method} just discontinued → set to 0")
                    ppl.method[i] = 0
                    ppl.on_contra[i] = False
                elif probs_for_year.get(int(new_method), 0.0) > 0.0:
                    print(f"    Switched to method {new_method} but it is also stocked out → set to 0")
                    ppl.method[i] = 0
                    ppl.on_contra[i] = False
                else:
                    print(f"    Agent {i} successfully switched to method {new_method}")
                    ppl.method[i] = new_method
                    ppl.on_contra[i] = True
