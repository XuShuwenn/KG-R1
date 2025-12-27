"""
KG Search module for modular knowledge graph retrieval.
"""

# NOTE(kgqa_agent mode): training uses the SPARQL bridge and only relies on
# `kg_r1.search.error_types.KGErrorType`. The original KG-R1 FastAPI server and
# action registry are legacy and intentionally not imported at package import
# time to avoid pulling in heavy deps (fastapi/uvicorn/pydantic) and to keep the
# import side-effect free.
#
# Legacy exports (kept for reference):
# from .actions import ActionType, ActionHandler, ACTION_REGISTRY, SearchRequest
# from .server import KnowledgeGraphRetriever, app

__all__ = []
