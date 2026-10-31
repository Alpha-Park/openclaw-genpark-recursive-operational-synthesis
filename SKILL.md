---
name: openclaw-genpark-recursive-operational-synthesis
description: OpenClaw GenPark Recursive Operational Synthesis - Bellman MDP trajectory policy and Lyapunov stability verifier with semantic token compression.
category: Service
version: 1.0.0
license: MIT
tools:
  - solve_mdp_value_iteration
  - compress_trajectory_tokens
  - calculate_lyapunov_stability
---

# Openclaw Genpark Recursive Operational Synthesis

## Overview & Agent Behavioral Directives
You are an autonomous agent specialized in **Openclaw Genpark Recursive Operational Synthesis**.
Your primary directive is to execute: OpenClaw GenPark Recursive Operational Synthesis - Bellman MDP trajectory policy and Lyapunov stability verifier with semantic token compression.
Always ensure mathematical precision, parameter validation, and deterministic outputs.

## When to Activate This Skill
- Activate when an agent workflow or user instruction requires:
- `solve_mdp_value_iteration`
- `compress_trajectory_tokens`
- `calculate_lyapunov_stability`
- Activate when operating in zero-dependency environments requiring pure Python standard library execution.

## How to Execute

### Option 1: Direct Python Execution (Zero-Dependency)
```python
from client import *

# Client operates purely on standard Python library:
# Initialized and ready for direct function invocation.
```

### Option 2: FastMCP Tool Calling
Tools registered in `mcp_server.py`:
- `solve_mdp_value_iteration`
- `compress_trajectory_tokens`
- `calculate_lyapunov_stability`

## Core Mathematical & Algorithmic Guardrails
- **Zero-Dependency Policy**: 100% Python standard library only (`math`, `heapq`, `hashlib`, `urllib`, `collections`, `itertools`, `typing`).
- **Zero-Mock Policy**: Pure deterministic algorithms with full validation and error bounds.
