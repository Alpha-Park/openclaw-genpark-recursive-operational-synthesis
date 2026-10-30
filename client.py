# -*- coding: utf-8 -*-
"""
OpenClaw GenPark Recursive Operational Synthesis - Bellman MDP Trajectory Policy & Lyapunov Stabilizer
Standard Library only: math, typing.
Calculates optimal Bellman policy via MDP value iteration, executes semantic prompt token compression,
and measures trajectory convergence using discrete Lyapunov exponent estimation.
"""

import math
from typing import Dict, List, Any

class OpenClawGenParkRecursiveOperationalSynthesis:
    def __init__(self, discount_factor: float = 0.90):
        self.gamma = discount_factor

    def solve_mdp_value_iteration(self, states: List[str],
                                  actions: List[str],
                                  transitions: Dict[str, Dict[str, Dict[str, float]]],
                                  rewards: Dict[str, Dict[str, float]],
                                  gamma: float = None,
                                  theta_conv: float = 1e-4) -> Dict[str, Any]:
        """
        Solves discrete Bellman optimality equation via value iteration:
        V(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s') ]
        """
        g = gamma if gamma is not None else self.gamma
        V = {s: 0.0 for s in states}
        policy = {s: actions[0] for s in states}

        iterations = 0
        max_iters = 100

        while iterations < max_iters:
            iterations += 1
            delta = 0.0
            new_V = {}

            for s in states:
                best_q = -1e9
                best_action = policy[s]

                for a in actions:
                    r = rewards.get(s, {}).get(a, 0.0)
                    expected_future = 0.0
                    next_states = transitions.get(s, {}).get(a, {})
                    for s_next, prob in next_states.items():
                        expected_future += prob * V.get(s_next, 0.0)

                    q_val = r + g * expected_future
                    if q_val > best_q:
                        best_q = q_val
                        best_action = a

                new_V[s] = best_q
                policy[s] = best_action
                delta = max(delta, abs(new_V[s] - V[s]))

            V = new_V
            if delta < theta_conv:
                break

        return {
            "iterations_converged": iterations,
            "discount_factor_gamma": g,
            "optimal_value_function": {s: round(v, 4) for s, v in V.items()},
            "optimal_policy": policy
        }

    def compress_trajectory_tokens(self, trajectory_log: str,
                                   target_token_budget: int = 100) -> Dict[str, Any]:
        """
        Compresses verbose LLM agent execution logs into deduplicated dense semantic summaries.
        """
        raw_words = trajectory_log.split()
        original_length = len(raw_words)

        # Filter out filler words
        stop_words = {"the", "a", "an", "and", "or", "to", "in", "of", "with", "for", "is", "was", "were", "been"}
        filtered = [w for w in raw_words if w.lower() not in stop_words]

        # Deduplicate sequential identical actions
        deduped = []
        for w in filtered:
            if not deduped or deduped[-1] != w:
                deduped.append(w)

        # Truncate to budget while preserving first 20% and last 80% of budget
        if len(deduped) > target_token_budget:
            head_len = int(target_token_budget * 0.3)
            tail_len = target_token_budget - head_len - 1
            compressed_words = deduped[:head_len] + ["...[TRUNCATED]..."] + deduped[-tail_len:]
        else:
            compressed_words = deduped

        compressed_text = " ".join(compressed_words)
        compression_ratio = len(compressed_words) / max(1, original_length)

        return {
            "original_token_count": original_length,
            "compressed_token_count": len(compressed_words),
            "compression_ratio": round(compression_ratio, 3),
            "space_savings_pct": round((1.0 - compression_ratio) * 100.0, 2),
            "compressed_trajectory": compressed_text
        }

    def calculate_lyapunov_stability(self, trajectory_deviations: List[float]) -> Dict[str, Any]:
        """
        Estimates the discrete largest Lyapunov exponent lambda across trajectory error deviations:
        lambda = (1 / N) * sum_{k=1}^N ln( |Delta x_{k+1}| / |Delta x_k| )
        If lambda < 0: trajectory exponentially converges to stable attractor.
        If lambda >= 0: chaotic divergence or unstable oscillation.
        """
        if len(trajectory_deviations) < 3:
            return {"error": "Need at least 3 deviations"}

        divergences = []
        for i in range(len(trajectory_deviations) - 1):
            d_curr = max(1e-6, abs(trajectory_deviations[i]))
            d_next = max(1e-6, abs(trajectory_deviations[i + 1]))
            ratio = d_next / d_curr
            divergences.append(math.log(ratio))

        lyapunov_exponent = sum(divergences) / len(divergences)

        if lyapunov_exponent < -0.10:
            regime = "EXPONENTIALLY_STABLE_CONVERGENT"
        elif lyapunov_exponent <= 0.05:
            regime = "MARGINALLY_STABLE_BOUNDED"
        else:
            regime = "CHAOTIC_DIVERGENT_UNSTABLE"

        return {
            "trajectory_steps_evaluated": len(trajectory_deviations),
            "lyapunov_exponent": round(lyapunov_exponent, 4),
            "stability_regime": regime,
            "is_trajectory_stable": lyapunov_exponent < 0.05
        }
