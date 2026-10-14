from claimguard.agent.graph import analyze_claim, build_graph
from claimguard.vlm.groq_backend import GroqVLM

graph = build_graph(GroqVLM())
print(analyze_claim(graph, "data/synthetic/claims/claim_001").model_dump_json(indent=2))