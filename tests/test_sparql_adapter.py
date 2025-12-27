import sys
import types
import unittest
from unittest.mock import patch

from kg_r1.kgqa_bridge.sparql_adapter import KGQASparqlAdapter


if "rank_bm25" not in sys.modules:
    class _DummyBM25:
        def __init__(self, docs):
            self._docs = docs

        def get_scores(self, *_):
            return [0.0 for _ in self._docs]

    sys.modules["rank_bm25"] = types.SimpleNamespace(BM25Okapi=_DummyBM25)


class _DummyDirectClient:
    def __init__(self, *_, **__):
        self._resolve_calls = []
        self._seen_relations_set = set()
        self._pending_flatten_relations = {}

    def clear_pending_flatten_relations(self, entity_ids=None):
        if entity_ids is None:
            self._pending_flatten_relations = {}
            return None
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        for eid in entity_ids:
            self._pending_flatten_relations.pop(eid, None)
        return None

    def _resolve_entity(self, entity: str) -> str:
        self._resolve_calls.append(entity)
        return f"m.resolved_{entity.replace(' ', '_')}"

    def get_relations(self, *_ , **__):
        relations = [{"relation": "people.person.parents"}]
        for rel in relations:
            self._seen_relations_set.add(rel["relation"])
        return relations

    def format_relations_for_prompt(self, relations):
        return "\n".join(rel["relation"] for rel in relations)

    def get_triples(self, *_ , **__):
        return {
            "triples": [
                {
                    "head": "Natalie Portman",
                    "head_id": "m.natalie",
                    "relation": "people.person.parents",
                    "tail": "Avner Hershlag",
                    "tail_id": "m.avner",
                }
            ]
        }

    def get_pending_flatten_relations(self, entity_id=None):
        if entity_id and entity_id in self._pending_flatten_relations:
            return self._pending_flatten_relations[entity_id]
        return []

    def rank_by_similarity(self, relations, question, key):
        return relations


class KGQASparqlAdapterTest(unittest.TestCase):
    def test_relations_and_triples_flow(self):
        with patch(
            "kgqa_agent.src.tools.direct_sparql_client.DirectSPARQLKGClient",
            _DummyDirectClient,
        ):
            adapter = KGQASparqlAdapter("http://localhost:8890/sparql", kg_top_k=2)

            text, payload = adapter.run_query("sample__idx_0", 'get_relations("Natalie Portman")')
            self.assertIn("people.person.parents", text)
            self.assertTrue(payload["success"])

            text, payload = adapter.run_query(
                "sample__idx_0",
                'get_triples("Natalie Portman", ["people.person.parents"])',
            )
            self.assertIn("Natalie Portman", text)
            self.assertTrue(payload["success"])

    def test_invalid_relation_rejected(self):
        with patch(
            "kgqa_agent.src.tools.direct_sparql_client.DirectSPARQLKGClient",
            _DummyDirectClient,
        ):
            adapter = KGQASparqlAdapter("http://localhost:8890/sparql", kg_top_k=1)
            adapter.run_query("sample__idx_1", 'get_relations("Natalie Portman")')

            text, payload = adapter.run_query(
                "sample__idx_1",
                'get_triples("Natalie Portman", ["film.actor.film"])',
            )
            self.assertIn("latest predicate list", text)
            self.assertFalse(payload["success"])

    def test_relation_filter_invoked(self):
        class _FilterDirectClient(_DummyDirectClient):
            def get_relations(self, *_ , **__):
                relations = [{"relation": f"rel_{i}"} for i in range(20)]
                for rel in relations:
                    self._seen_relations_set.add(rel["relation"])
                return relations

        class _StubModelClient:
            calls = 0

            def __init__(self, cfg):
                self.cfg = cfg

            def generate(self, prompt, **kwargs):
                _StubModelClient.calls += 1
                return '["rel_5", "rel_3", "rel_1"]'

        with patch(
            "kgqa_agent.src.tools.direct_sparql_client.DirectSPARQLKGClient",
            _FilterDirectClient,
        ), patch(
            "kg_r1.kgqa_bridge.sparql_adapter.BaseModelClient",
            _StubModelClient,
        ):
            _StubModelClient.calls = 0
            adapter = KGQASparqlAdapter(
                "http://localhost:8890/sparql",
                kg_top_k=3,
                relation_filter_model="dummy-model",
            )
            text, _ = adapter.run_query("sample__idx_2", 'get_relations("Natalie Portman")')
            self.assertIn("rel_5", text.splitlines()[0])
            self.assertGreater(_StubModelClient.calls, 0)


if __name__ == "__main__":
    unittest.main()


