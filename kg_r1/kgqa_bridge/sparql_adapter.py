import json
import logging
import random
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kg_r1.search.error_types import KGErrorType

try:
    from kgqa_agent.src.tools.relation_normalizer import normalize_relation
    from kgqa_agent.src.eval.model_client import BaseModelClient, ModelConfig
    from kgqa_agent.prompts.filter_prompt import __doc__ as filter_prompt_template
    from kgqa_agent.prompts.flatten_rel_filter_prompt import (
        __doc__ as flatten_rel_filter_prompt_template,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "kgqa_agent package is required for KGQASparqlAdapter. "
        "Ensure the repo submodule is available and on PYTHONPATH."
    ) from exc


@dataclass
class _SessionState:
    """Per-sample interaction state to mimic kgqa_agent's conversational memory."""

    entity_registry: Dict[str, str] = field(default_factory=dict)
    seen_relations: set[str] = field(default_factory=set)
    last_relations_text: str = ""
    last_entities_text: str = ""
    initial_entity_names: List[str] = field(default_factory=list)
    query_count: int = 0  # Track number of KG queries executed for this sample
    trace: List[Dict[str, Any]] = field(default_factory=list)


class KGQASparqlAdapter:
    """
    Lightweight bridge that reuses kgqa_agent's DirectSPARQLKGClient to execute
    get_relations / get_triples queries issued inside <kg-query> blocks during RL training.
    """

    _GET_RELATIONS_PATTERN = re.compile(r'get_relations\s*\(\s*(["\'])(.*?)\1\s*\)', re.DOTALL)
    _GET_TRIPLES_PATTERN = re.compile(
        r'get_triples\s*\(\s*(["\'])(.*?)\1\s*,\s*\[(.*?)\]\s*\)', re.DOTALL
    )

    def __init__(
        self,
        sparql_endpoint: str,
        *,
        timeout: int = 10,
        kg_top_k: int = 10,
        max_calls: int = 10,
        relation_filter_model: Optional[str] = None,
        relation_filter_max_tokens: int = 1024,
        relation_filter_temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._sparql_endpoint = self._normalize_endpoint(sparql_endpoint)
        self._timeout = timeout
        self._kg_top_k = kg_top_k
        self._max_calls = max_calls
        self._logger = logger or logging.getLogger(__name__)
        self._client = None
        self._sessions: Dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._relation_filter_model = relation_filter_model
        self._relation_filter_max_tokens = relation_filter_max_tokens
        self._relation_filter_temperature = relation_filter_temperature
        self._filter_client: Optional[BaseModelClient] = None
        self._filter_total_calls = 0
        self._filter_success_count = 0
        self._filter_fallback_count = 0
        self._filter_parse_fail_count = 0
        self._filter_last_log_time = 0.0

    @staticmethod
    def _normalize_endpoint(sparql_endpoint: str) -> str:
        endpoint = sparql_endpoint.strip()
        if not endpoint.endswith("/sparql"):
            endpoint = f"{endpoint.rstrip('/')}/sparql"
        return endpoint

    def _ensure_client(self):
        if self._client is None:
            from kgqa_agent.src.tools.direct_sparql_client import DirectSPARQLKGClient

            self._client = DirectSPARQLKGClient(
                sparql_endpoint=self._sparql_endpoint,
                timeout=self._timeout,
                llm_filter_callback=self._llm_filter_callback
                if self._relation_filter_model
                else None,
            )
            if self._relation_filter_model:
                self._logger.info(
                    "[KG_FILTER] Enabled relation filter model: %s",
                    self._relation_filter_model,
                )

    def _ensure_filter_client(self) -> None:
        if not self._relation_filter_model or self._filter_client is not None:
            return
        if BaseModelClient is None:
            self._logger.warning(
                "[KG_FILTER] BaseModelClient is unavailable; "
                "relation_filter_model=%s will be ignored.",
                self._relation_filter_model,
            )
            return
        try:
            self._filter_client = BaseModelClient(
                ModelConfig(
                    model=self._relation_filter_model,
                    temperature=self._relation_filter_temperature,
                    max_tokens=self._relation_filter_max_tokens,
                    timeout=self._timeout,
                )
            )
        except Exception as exc:  # pragma: no cover - network/SDK errors
            self._logger.warning(
                "[KG_FILTER] Failed to initialize relation filter model %s: %s",
                self._relation_filter_model,
                exc,
            )
            self._filter_client = None

    def reset(self, sample_id: Optional[str] = None) -> None:
        """Clear adapter state. If sample_id is None, reset all sessions."""
        with self._sessions_lock:
            if sample_id is None:
                self._sessions.clear()
            else:
                session = self._sessions.pop(sample_id, None)
                # Query count is reset when session is removed
        if self._client:
            # Clear flatten relations cache between questions to mimic kgqa_agent behavior
            self._client.clear_pending_flatten_relations()
        if sample_id is None:
            for session in self._sessions.values():
                session.trace.clear()

    @staticmethod
    def _append_trace(session: _SessionState, entry: Dict[str, Any]) -> None:
        session.trace.append(entry)
        # Keep trace list bounded
        if len(session.trace) > 200:
            del session.trace[:-200]

    @staticmethod
    def _get_session_trace(session: _SessionState) -> List[Dict[str, Any]]:
        if not session.trace:
            return []
        try:
            return json.loads(json.dumps(session.trace))
        except Exception:  # pragma: no cover - serialization guard
            return session.trace[-50:]

    def run_query(
        self,
        sample_id: str,
        query_str: str,
        *,
        question: str = "",
        topic_entities: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute a kgqa-style query and return formatted text plus raw payload."""
        if not query_str:
            return self._format_error(
                "Empty query string received. Use get_relations(entity) or get_triples(entity, [relations]).",
                sample_id,
                KGErrorType.FORMAT_ERROR,
            )

        session = self._get_session(sample_id, topic_entities)
        
        # Check query count limit (mimicking kgqa_agent's max_calls behavior)
        if session.query_count >= self._max_calls:
            try:
                from verl.trainer.ppo.prompts import FORCE_ANSWER_PROMPT
            except ImportError:
                FORCE_ANSWER_PROMPT = (
                    "You have reached the maximum number of queries. "
                    "Based on the information gathered, provide your final answer in <answer> tags. "
                    "Strict Format: <answer>[\"Answer1\", \"Answer2\"]</answer>. "
                    "The answer(s) must be concise entity names copied exactly from the KG results."
                )
            
            self._logger.info(
                f"Max calls ({self._max_calls}) reached for sample {sample_id}. "
                f"Returning force answer prompt."
            )
            return (
                FORCE_ANSWER_PROMPT,
                self._build_payload(
                    success=False,
                    content=FORCE_ANSWER_PROMPT,
                    sample_id=sample_id,
                    meta={
                        "action": "max_calls_reached",
                        "query_count": session.query_count,
                        "max_calls": self._max_calls,
                    },
                ),
            )
        
        # Increment query count before executing query
        session.query_count += 1
        
        self._ensure_client()
        augmented_question = self._augment_question(question, session)

        relations_match = self._GET_RELATIONS_PATTERN.match(query_str.strip())
        if relations_match:
            entity = relations_match.group(2).strip()
            return self._handle_get_relations(
                sample_id=sample_id,
                entity=entity,
                question=augmented_question,
                session=session,
            )

        triples_match = self._GET_TRIPLES_PATTERN.match(query_str.strip())
        if triples_match:
            entity = triples_match.group(2).strip()
            relations = self._parse_relations(triples_match.group(3))
            return self._handle_get_triples(
                sample_id=sample_id,
                entity=entity,
                relations=relations,
                question=augmented_question,
                session=session,
            )

        return self._format_error(
            f"[Could not parse query: {query_str}]",
            sample_id,
            KGErrorType.FORMAT_ERROR,
        )

    def _get_session(
        self,
        sample_id: str,
        topic_entities: Optional[Dict[str, str]] = None,
    ) -> _SessionState:
        with self._sessions_lock:
            session = self._sessions.get(sample_id)
            if session is None:
                session = _SessionState()
                self._sessions[sample_id] = session
            if topic_entities:
                self._seed_registry_from_topic_entities(session, topic_entities)
                session.initial_entity_names = self._extract_initial_entity_names(topic_entities)
            return session

    def _handle_get_relations(
        self,
        *,
        sample_id: str,
        entity: str,
        question: str,
        session: _SessionState,
    ) -> Tuple[str, Dict[str, Any]]:
        import time
        start_time = time.time()
        self._logger.info(f"[KG_QUERY] Starting get_relations for entity: {entity}")
        
        entity_resolved = self._resolve_and_register_entity(entity, session)
        resolve_time = time.time() - start_time
        self._logger.info(f"[KG_QUERY] Entity resolution took {resolve_time:.2f}s: {entity} -> {entity_resolved}")
        
        if not entity_resolved:
            return self._entity_error(entity, session, sample_id)

        try:
            query_start = time.time()
            self._logger.info(f"[KG_QUERY] Calling _client.get_relations for {entity_resolved}...")
            relations = self._client.get_relations(
                entity_resolved,
                question=question,
                top_k=self._kg_top_k * 3,
            )
            query_time = time.time() - query_start
            self._logger.info(f"[KG_QUERY] get_relations completed in {query_time:.2f}s, got {len(relations) if relations else 0} relations")
        except Exception as exc:  # pragma: no cover - network errors
            # Follow kgqa_agent pattern: log error but return simple message to model
            self._logger.warning("SPARQL get_relations failed: %s", exc)
            formatted = "No relations found."
            payload = self._build_payload(
                success=False,
                content=formatted,
                sample_id=sample_id,
                meta={
                    "action": "get_relations",
                    "error_type": KGErrorType.SERVER_ERROR,
                },
            )
            return formatted, payload

        if not relations:
            formatted = "No relations found."
        else:
            # Align with kgqa_agent: shuffle if > 10 to avoid position bias
            if len(relations) > 10:
                random.shuffle(relations)
                # Filter with LLM if filter model is configured
                if self._relation_filter_model:
                    relations = self._filter_relations_with_llm(
                        relations,
                        question,
                        entity,
                        use_flatten_prompt=False,
                    )
            relations = relations[: self._kg_top_k]
            formatted = self._client.format_relations_for_prompt(relations)
            for rel in relations:
                rel_name = rel.get("relation")
                if rel_name:
                    session.seen_relations.add(normalize_relation(rel_name))

        session.last_relations_text = formatted
        session.last_entities_text = ""

        trace_entry = {
            "type": "tool_call",
            "tool": "get_relations",
            "args": {"entity": entity, "entity_resolved": entity_resolved},
            "result_count": len(relations) if relations else 0,
            "tool_output": formatted,
        }
        self._append_trace(session, trace_entry)

        payload = self._build_payload(
            success=True,
            content=formatted,
            sample_id=sample_id,
            meta={
                "entity": entity_resolved,
                "action": "get_relations",
                "trace": self._get_session_trace(session),
            },
        )
        return formatted, payload

    def _handle_get_triples(
        self,
        *,
        sample_id: str,
        entity: str,
        relations: List[str],
        question: str,
        session: _SessionState,
    ) -> Tuple[str, Dict[str, Any]]:
        entity_resolved = self._resolve_and_register_entity(entity, session)
        if not entity_resolved:
            return self._entity_error(entity, session, sample_id)

        if session.seen_relations:
            for rel in relations:
                norm_rel = normalize_relation(rel)
                if norm_rel not in session.seen_relations:
                    return self._relation_choice_error(rel, session, sample_id)

        try:
            result = self._client.get_triples(
                entity_resolved,
                relations,
                limit_per_relation=5,
                question=question,
                return_with_cvt_info=True,
            )
            if isinstance(result, dict):
                triples = result.get("triples", [])
                cvt_info = result.get("cvt_info")
            else:
                triples = result
                cvt_info = None
        except Exception as exc:  # pragma: no cover - network errors
            # Follow kgqa_agent pattern: log error but return simple message to model
            self._logger.warning("SPARQL get_triples failed: %s", exc)
            formatted = "No triples found."
            payload = self._build_payload(
                success=False,
                content=formatted,
                sample_id=sample_id,
                meta={
                    "action": "get_triples",
                    "error_type": KGErrorType.SERVER_ERROR,
                },
            )
            return formatted, payload

        if not triples:
            formatted = "No triples found."
        else:
            formatted = "\n".join(
                f"[{triple.get('head')}, {triple.get('relation')}, {triple.get('tail')}]"
                for triple in triples
            )
            self._register_entities(triples, session)

        flatten_relations = self._client.get_pending_flatten_relations(entity_resolved)
        formatted_cvt_info: Optional[List[Dict[str, Any]]] = None
        if cvt_info:
            formatted_cvt_info = []
            for cvt in cvt_info:
                flattened_triples = []
                for triple in cvt.get("flattened_triples", []):
                    flattened_triples.append(
                        {
                            "head": triple.get("head"),
                            "head_id": triple.get("head_id"),
                            "relation": triple.get("relation"),
                            "tail": triple.get("tail"),
                            "tail_id": triple.get("tail_id"),
                        }
                    )
                formatted_cvt_info.append(
                    {
                        "original_relation": cvt.get("original_relation"),
                        "cvt_node_id": cvt.get("cvt_node_id"),
                        "cvt_relation": cvt.get("cvt_relation"),
                        "flatten_relation": cvt.get("flatten_relation"),
                        "direction": cvt.get("direction"),
                        "flattened_triples": flattened_triples,
                    }
                )

        if flatten_relations:
            flatten_rel_dicts = [{"relation": rel} for rel in flatten_relations]
            other_relations = [{"relation": rel} for rel in relations[1:]]
            combined_relations = other_relations + flatten_rel_dicts
            if (
                question
                and combined_relations
                and hasattr(self._client, "rank_by_similarity")
            ):
                try:
                    combined_relations = self._client.rank_by_similarity(
                        combined_relations, question, "relation"
                    )
                except Exception:
                    pass
            if len(combined_relations) > 10:
                random.shuffle(combined_relations)
                combined_relations = self._filter_relations_with_llm(
                    combined_relations, question, entity, use_flatten_prompt=False
                )
            for rel_dict in combined_relations:
                rel_name = rel_dict.get("relation")
                if rel_name:
                    session.seen_relations.add(normalize_relation(rel_name))

        session.last_entities_text = formatted
        session.last_relations_text = ""

        trace_entry: Dict[str, Any] = {
            "type": "tool_call",
            "tool": "get_triples",
            "args": {
                "entity": entity,
                "entity_resolved": entity_resolved,
                "relations": relations,
            },
            "result_count": len(triples),
            "tool_output": formatted,
        }
        if flatten_relations:
            trace_entry["flatten_relations"] = flatten_relations
        if formatted_cvt_info:
            trace_entry["cvt_info"] = formatted_cvt_info
        self._append_trace(session, trace_entry)

        payload = self._build_payload(
            success=True,
            content=formatted,
            sample_id=sample_id,
            meta={
                "entity": entity_resolved,
                "relations": relations,
                "action": "get_triples",
                "trace": self._get_session_trace(session),
            },
        )
        return formatted, payload

    def _llm_filter_callback(
        self,
        relations: List[Dict[str, str]],
        question: str,
        entity_name: str,
        use_flatten_prompt: bool = False,
    ) -> List[Dict[str, str]]:
        return self._filter_relations_with_llm(
            relations,
            question,
            entity_name,
            use_flatten_prompt=use_flatten_prompt,
        )

    def _filter_relations_with_llm(
        self,
        relations: List[Dict[str, str]],
        question: str,
        entity_name: str,
        *,
        use_flatten_prompt: bool = False,
    ) -> List[Dict[str, str]]:
        """Filter relations via LLM to mimic kgqa_agent behaviour."""
        if not self._relation_filter_model or not relations:
            return relations

        # Keep only entries that actually contain relation names
        relation_entries = [rel for rel in relations if rel.get("relation")]
        if len(relation_entries) <= 10 and not use_flatten_prompt:
            return relation_entries

        self._ensure_filter_client()
        if self._filter_client is None:
            # Failed to build client; fall back immediately
            return relation_entries

        # Shuffle before building prompt to avoid positional bias
        shuffled = relation_entries[:]
        random.shuffle(shuffled)

        prompt_template = (
            flatten_rel_filter_prompt_template
            if use_flatten_prompt
            else filter_prompt_template
        ) or ""
        prompt = (
            f"{prompt_template}\n"
            f"Question: {question}\n"
            f"Topic Entity: [\"{entity_name}\"]\n"
            f"Relations: {json.dumps([rel['relation'] for rel in shuffled])}\n\n"
            f"Your Selections: "
        )

        max_retries = 3
        last_exception: Optional[Exception] = None
        json_pattern = re.compile(r"\[.*?\]", re.DOTALL)

        for attempt in range(max_retries):
            self._filter_total_calls += 1
            try:
                response = self._filter_client.generate(
                    prompt,
                    temperature=self._relation_filter_temperature,
                    max_tokens=self._relation_filter_max_tokens,
                )
                json_match = json_pattern.search(response or "")
                if not json_match:
                    self._filter_parse_fail_count += 1
                    break

                parsed = json.loads(json_match.group(0))
                if not isinstance(parsed, list):
                    self._filter_parse_fail_count += 1
                    break

                rel_map = {rel["relation"]: rel for rel in relation_entries}
                ordered_filtered = [
                    rel_map[name]
                    for name in parsed
                    if isinstance(name, str) and name in rel_map
                ]
                if ordered_filtered:
                    self._filter_success_count += 1
                    self._maybe_log_filter_stats()
                    return ordered_filtered

                # If parsed successfully but no overlap, treat as fallback
                self._filter_fallback_count += 1
                break
            except Exception as exc:  # pragma: no cover - network/LLM errors
                last_exception = exc
                if attempt + 1 < max_retries:
                    time.sleep(min(4.0, 1.5 ** attempt))
                    continue
                self._filter_fallback_count += 1
                self._logger.debug(
                    "[KG_FILTER] LLM filter failed after %d attempts: %s",
                    attempt + 1,
                    exc,
                )

        self._maybe_log_filter_stats()
        return relation_entries

    def _maybe_log_filter_stats(self) -> None:
        now = time.time()
        # Log at most once per minute to avoid spam
        if now - self._filter_last_log_time < 60:
            return
        self._filter_last_log_time = now
        if not self._relation_filter_model:
            return
        self._logger.info(
            "[KG_FILTER] model=%s total=%d success=%d fallback=%d parse_fail=%d",
            self._relation_filter_model,
            self._filter_total_calls,
            self._filter_success_count,
            self._filter_fallback_count,
            self._filter_parse_fail_count,
        )

    @staticmethod
    def _parse_relations(relations_str: str) -> List[str]:
        parts = []
        for raw in relations_str.split(","):
            candidate = raw.strip()
            if not candidate:
                continue
            if (candidate.startswith('"') and candidate.endswith('"')) or (
                candidate.startswith("'") and candidate.endswith("'")
            ):
                candidate = candidate[1:-1]
            if candidate and candidate not in parts:
                parts.append(candidate)
        return parts[:4]

    def _entity_error(
        self,
        entity: str,
        session: _SessionState,
        sample_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        msg = (
            "Invalid entity. Use an entity returned by the previous step and copy it exactly."
        )
        if session.last_entities_text:
            msg += f"\n\nLast entities (names only):\n\n{session.last_entities_text}"
        return self._format_error(msg, sample_id, KGErrorType.FORMAT_ERROR)

    def _relation_choice_error(
        self,
        relation: str,
        session: _SessionState,
        sample_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        msg = (
            f"The relation '{relation}' is not in the latest predicate list. "
            "Choose predicates from the list below:\n\n"
            f"{session.last_relations_text}"
        )
        return self._format_error(msg, sample_id, KGErrorType.FORMAT_ERROR)

    def _resolve_and_register_entity(
        self,
        entity: str,
        session: _SessionState,
    ) -> Optional[str]:
        if not entity:
            return None
        if entity.startswith(("m.", "g.", "en.")):
            session.entity_registry[entity] = entity
            return entity

        key = entity.lower()
        if key in session.entity_registry:
            return session.entity_registry[key]

        resolved = self._client._resolve_entity(entity)
        if resolved:
            session.entity_registry[key] = resolved
            session.entity_registry[resolved] = entity
        return resolved

    @staticmethod
    def _register_entities(triples: List[Dict[str, Any]], session: _SessionState) -> None:
        for triple in triples:
            head_id = triple.get("head_id")
            head_name = triple.get("head")
            tail_id = triple.get("tail_id")
            tail_name = triple.get("tail")
            if head_id and head_name:
                session.entity_registry[head_name.lower()] = head_id
                session.entity_registry[head_id] = head_name
            if tail_id and tail_name:
                session.entity_registry[tail_name.lower()] = tail_id
                session.entity_registry[tail_id] = tail_name

    @staticmethod
    def _seed_registry_from_topic_entities(
        session: _SessionState,
        topic_entities: Dict[str, str],
    ) -> None:
        for key, value in topic_entities.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            if key.startswith(("m.", "g.", "en.")):
                session.entity_registry[value.lower()] = key
                session.entity_registry[key] = value
            elif value.startswith(("m.", "g.", "en.")):
                session.entity_registry[key.lower()] = value
                session.entity_registry[value] = key

    @staticmethod
    def _extract_initial_entity_names(topic_entities: Dict[str, str]) -> List[str]:
        names: List[str] = []
        for key, value in topic_entities.items():
            if isinstance(key, str) and key.startswith(("m.", "g.", "en.")) and isinstance(value, str):
                names.append(value.strip())
            elif isinstance(value, str) and value.startswith(("m.", "g.", "en.")) and isinstance(key, str):
                names.append(key.strip())
        seen = set()
        ordered = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    @staticmethod
    def _augment_question(question: str, session: _SessionState) -> str:
        if not session.initial_entity_names:
            return question or ""
        suffix = " ".join(session.initial_entity_names)
        return (question + " " + suffix).strip()

    def _build_payload(
        self,
        *,
        success: bool,
        content: str,
        sample_id: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "object": "kg_retrieval",
            "success": success,
            "choices": [{"message": {"role": "tool", "content": content}}],
            "kg_metadata": {
                "success": success,
                "error_type": KGErrorType.SUCCESS if success else KGErrorType.SERVER_ERROR,
            },
            "request_payload": {
                "sample_id": sample_id,
                "sparql_endpoint": self._sparql_endpoint,
                **meta,
            },
        }

    def _format_error(
        self,
        message: str,
        sample_id: str,
        error_type: KGErrorType,
    ) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "object": "kg_retrieval",
            "success": False,
            "choices": [{"message": {"role": "tool", "content": message}}],
            "kg_metadata": {"success": False, "error_type": error_type},
            "request_payload": {
                "sample_id": sample_id,
                "sparql_endpoint": self._sparql_endpoint,
            },
        }
        return message, payload


