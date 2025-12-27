# NOTE(kgqa_agent mode): training uses SPARQL bridge and does not use the legacy
# MultiTQ RoG-subgraph FastAPI stack. Avoid importing heavy deps at package import
# time; keep legacy exports commented out for reference.
#
# from .knowledge_graph_multitq import KnowledgeGraphMultiTQ
# from .server_multitq import KnowledgeGraphRetrieverMultiTQ
# from .actions_multitq import *

__all__ = []