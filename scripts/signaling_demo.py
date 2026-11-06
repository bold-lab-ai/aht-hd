"""CPU-only signaling-game numerical illustration, without human data."""

import json

from ah2ac2.theory.signaling import effective_reference, global_maximizers

p, eta, penalty = 0.7, 0.2, 0.1
q = effective_reference(p, eta)
optima = global_maximizers(q, penalty)
assert all(t > q > 0.5 for t in optima)
print(
    json.dumps(
        {
            "clean_majority": p,
            "noise_rate": eta,
            "pooled_reference": q,
            "kl_penalty": penalty,
            "global_optima": optima,
        },
        indent=2,
    )
)
