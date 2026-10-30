# -*- coding: utf-8 -*-
from client import OpenClawGenParkRecursiveOperationalSynthesis

synth = OpenClawGenParkRecursiveOperationalSynthesis()

# 1. Bellman MDP Policy
states = ["INIT", "TOOL_FAIL", "SUCCESS", "ABORT"]
actions = ["RETRY", "ESCALATE", "TERMINATE"]
transitions = {
    "INIT": {"RETRY": {"SUCCESS": 0.8, "TOOL_FAIL": 0.2}, "ESCALATE": {"SUCCESS": 0.95, "TOOL_FAIL": 0.05}, "TERMINATE": {"ABORT": 1.0}},
    "TOOL_FAIL": {"RETRY": {"SUCCESS": 0.5, "TOOL_FAIL": 0.5}, "ESCALATE": {"SUCCESS": 0.9, "ABORT": 0.1}, "TERMINATE": {"ABORT": 1.0}},
    "SUCCESS": {"RETRY": {"SUCCESS": 1.0}, "ESCALATE": {"SUCCESS": 1.0}, "TERMINATE": {"SUCCESS": 1.0}},
    "ABORT": {"RETRY": {"ABORT": 1.0}, "ESCALATE": {"ABORT": 1.0}, "TERMINATE": {"ABORT": 1.0}}
}
rewards = {
    "INIT": {"RETRY": 5.0, "ESCALATE": 2.0, "TERMINATE": -10.0},
    "TOOL_FAIL": {"RETRY": -1.0, "ESCALATE": 4.0, "TERMINATE": -10.0},
    "SUCCESS": {"RETRY": 10.0, "ESCALATE": 10.0, "TERMINATE": 10.0},
    "ABORT": {"RETRY": 0.0, "ESCALATE": 0.0, "TERMINATE": 0.0}
}
mdp = synth.solve_mdp_value_iteration(states, actions, transitions, rewards, gamma=0.9)
print(f"MDP Convergence: {mdp['iterations_converged']} iterations")
print(f"  Optimal Policy: {mdp['optimal_policy']}")

# 2. Token Compression
raw_log = "The agent attempted to query the database and the query succeeded. The agent then proceeded to process the results in the loop and verify the outputs with high confidence."
comp = synth.compress_trajectory_tokens(raw_log, target_token_budget=12)
print(f"\nToken Compression: {comp['space_savings_pct']}% saved -> '{comp['compressed_trajectory']}'")

# 3. Lyapunov Stability
devs = [1.2, 0.85, 0.45, 0.22, 0.09, 0.03]
lyap = synth.calculate_lyapunov_stability(devs)
print(f"\nLyapunov Stability: Exponent = {lyap['lyapunov_exponent']} ({lyap['stability_regime']})")
