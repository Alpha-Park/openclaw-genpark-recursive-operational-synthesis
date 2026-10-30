# -*- coding: utf-8 -*-
import sys, json
from client import OpenClawGenParkRecursiveOperationalSynthesis

client = OpenClawGenParkRecursiveOperationalSynthesis()

TOOLS = [
    {
        "name": "solve_mdp_value_iteration",
        "description": "Computes optimal Bellman policy across agent execution states via MDP value iteration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "states": {"type": "array", "items": {"type": "string"}},
                "actions": {"type": "array", "items": {"type": "string"}},
                "transitions": {"type": "object"},
                "rewards": {"type": "object"},
                "gamma": {"type": "number"}
            },
            "required": ["states", "actions", "transitions", "rewards"]
        }
    },
    {
        "name": "compress_trajectory_tokens",
        "description": "Compresses verbose LLM agent execution logs into deduplicated dense semantic summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trajectory_log": {"type": "string"},
                "target_token_budget": {"type": "integer"}
            },
            "required": ["trajectory_log"]
        }
    },
    {
        "name": "calculate_lyapunov_stability",
        "description": "Estimates discrete largest Lyapunov exponent to verify agent trajectory stability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trajectory_deviations": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["trajectory_deviations"]
        }
    }
]

def handle_request(req):
    method = req.get("method")
    rid = req.get("id")
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        params = req.get("params", {})
        tname = params.get("name")
        args = params.get("arguments", {})
        if tname == "solve_mdp_value_iteration":
            res = client.solve_mdp_value_iteration(args["states"], args["actions"], args["transitions"], args["rewards"], args.get("gamma"))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        elif tname == "compress_trajectory_tokens":
            res = client.compress_trajectory_tokens(args["trajectory_log"], args.get("target_token_budget", 100))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        elif tname == "calculate_lyapunov_stability":
            res = client.calculate_lyapunov_stability(args["trajectory_deviations"])
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "Invalid Request"}}

def main():
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
