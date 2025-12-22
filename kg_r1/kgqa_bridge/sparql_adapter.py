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
        # Ensure filter model is always configured (aligning with kgqa_agent behavior)
        # If not provided, use default "gpt-4o-mini" like kgqa_agent
        self._relation_filter_model = relation_filter_model or "gpt-4o-mini"
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

            # Filter model is always configured (defaults to "gpt-4o-mini" if not provided)
            self._client = DirectSPARQLKGClient(
                sparql_endpoint=self._sparql_endpoint,
                timeout=self._timeout,
                llm_filter_callback=self._llm_filter_callback,
            )
            import os
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", None)
            self._logger.info(
                "[KG_FILTER] Enabled relation filter model: %s (api_key_present=%s, base_url=%s)",
                self._relation_filter_model,
                bool(api_key),
                base_url or "default OpenAI API",
            )

    def _ensure_filter_client(self) -> None:
        # Filter model is always configured (defaults to "gpt-4o-mini" if not provided)
        if self._filter_client is not None:
            return
        if BaseModelClient is None:
            self._logger.warning(
                "[KG_FILTER] BaseModelClient is unavailable; "
                "relation_filter_model=%s will be ignored.",
                self._relation_filter_model,
            )
            return
        
        # Check environment variables
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", None)
        
        self._logger.info(
            f"[KG_FILTER] Initializing filter client: model={self._relation_filter_model}, "
            f"api_key_present={bool(api_key)}, base_url={base_url or 'default OpenAI API'}"
        )
        
        try:
            self._filter_client = BaseModelClient(
                ModelConfig(
                    model=self._relation_filter_model,
                    temperature=self._relation_filter_temperature,
                    max_tokens=self._relation_filter_max_tokens,
                    timeout=self._timeout,
                )
            )
            self._logger.info(
                f"[KG_FILTER] Successfully initialized filter client for model {self._relation_filter_model}"
            )
        except Exception as exc:  # pragma: no cover - network/SDK errors
            self._logger.warning(
                "[KG_FILTER] Failed to initialize relation filter model %s: %s",
                self._relation_filter_model,
                exc,
                exc_info=True,  # Include full traceback
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
        import time
        total_start = time.time()
        
        if not query_str:
            return self._format_error(
                "Empty query string received. Use get_relations(entity) or get_triples(entity, [relations]).",
                sample_id,
                KGErrorType.FORMAT_ERROR,
            )

        session_start = time.time()
        session = self._get_session(sample_id, topic_entities)
        session_time = time.time() - session_start
        if session_time > 0.01:
            self._logger.info(f"[TIMING] run_query: session setup took {session_time:.3f}s")
        
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
        
        ensure_start = time.time()
        self._ensure_client()
        ensure_time = time.time() - ensure_start
        if ensure_time > 0.01:
            self._logger.info(f"[TIMING] run_query: _ensure_client took {ensure_time:.3f}s")
        
        augment_start = time.time()
        augmented_question = self._augment_question(question, session)
        augment_time = time.time() - augment_start
        if augment_time > 0.01:
            self._logger.info(f"[TIMING] run_query: _augment_question took {augment_time:.3f}s")

        parse_start = time.time()
        relations_match = self._GET_RELATIONS_PATTERN.match(query_str.strip())
        triples_match = self._GET_TRIPLES_PATTERN.match(query_str.strip())
        parse_time = time.time() - parse_start
        if parse_time > 0.01:
            self._logger.info(f"[TIMING] run_query: query parsing took {parse_time:.3f}s")
        
        if relations_match:
            entity = relations_match.group(2).strip()
            result = self._handle_get_relations(
                sample_id=sample_id,
                entity=entity,
                question=augmented_question,
                session=session,
            )
            total_time = time.time() - total_start
            if total_time > 1.0:
                self._logger.info(f"[TIMING] run_query: TOTAL get_relations took {total_time:.3f}s")
            return result

        if triples_match:
            entity = triples_match.group(2).strip()
            parse_rel_start = time.time()
            relations = self._parse_relations(triples_match.group(3))
            parse_rel_time = time.time() - parse_rel_start
            if parse_rel_time > 0.01:
                self._logger.info(f"[TIMING] run_query: _parse_relations took {parse_rel_time:.3f}s")
            result = self._handle_get_triples(
                sample_id=sample_id,
                entity=entity,
                relations=relations,
                question=augmented_question,
                session=session,
            )
            total_time = time.time() - total_start
            if total_time > 1.0:
                self._logger.info(f"[TIMING] run_query: TOTAL get_triples took {total_time:.3f}s")
            return result

        # Use clearer error message that's less likely to be mistaken as entity name
        error_msg = (
            "Invalid query format. Use get_relations(\"entity_name\") "
            "or get_triples(\"entity_name\", [\"relation1\", ...])."
        )
        return self._format_error(error_msg, sample_id, KGErrorType.FORMAT_ERROR)

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
        total_start = time.time()
        self._logger.info(f"[KG_QUERY] Starting get_relations for entity: {entity}")
        
        # Augment question with initial entity names (aligning with kgqa_agent)
        augmented_question = self._augment_question(question, session)
        
        resolve_start = time.time()
        entity_resolved = self._resolve_and_register_entity(entity, session)
        resolve_time = time.time() - resolve_start
        if resolve_time > 0.1:
            self._logger.info(f"[TIMING] _handle_get_relations: entity resolution took {resolve_time:.3f}s: {entity} -> {entity_resolved}")
        
        if not entity_resolved:
            return self._entity_error(entity, session, sample_id)

        try:
            query_start = time.time()
            self._logger.info(f"[KG_QUERY] Calling _client.get_relations for {entity_resolved}...")
            relations = self._client.get_relations(
                entity_resolved,
                question=augmented_question,  # Use augmented_question (aligning with kgqa_agent)
                top_k=self._kg_top_k * 3,
            )
            query_time = time.time() - query_start
            if query_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_relations: _client.get_relations took {query_time:.3f}s, got {len(relations) if relations else 0} relations")
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
            process_start = time.time()
            # Always apply LLM filter regardless of relation count (to ensure filter model is used)
            # Shuffle to avoid position bias before filtering
            shuffle_start = time.time()
            random.shuffle(relations)
            shuffle_time = time.time() - shuffle_start
            if shuffle_time > 0.01:
                self._logger.info(f"[TIMING] _handle_get_relations: shuffle took {shuffle_time:.3f}s")
            # Filter with LLM (filter model is always configured)
            filter_start = time.time()
            self._logger.info(f"[KG_FILTER] Applying LLM filter to {len(relations)} relations (entity: {entity})")
            relations = self._filter_relations_with_llm(
                relations,
                question,
                entity,
                use_flatten_prompt=False,
            )
            filter_time = time.time() - filter_start
            if filter_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_relations: LLM filtering took {filter_time:.3f}s")
            self._logger.info(f"[KG_FILTER] After filtering: {len(relations)} relations remaining")
            truncate_start = time.time()
            relations = relations[: self._kg_top_k]
            format_start = time.time()
            formatted = self._client.format_relations_for_prompt(relations)
            format_time = time.time() - format_start
            if format_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_relations: format_relations_for_prompt took {format_time:.3f}s")
            process_time = time.time() - process_start
            if process_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_relations: post-processing took {process_time:.3f}s")
            # Update seen_relations without normalize (aligning with kgqa_agent behavior)
            # Note: kgqa_agent uses normalize_relation in get_triples validation, but not in get_relations seen_relations update
            for rel in relations:
                rel_name = rel.get("relation")
                if rel_name:
                    session.seen_relations.add(rel_name)  # Don't normalize, matching kgqa_agent line 150

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
        import time
        total_start = time.time()
        self._logger.info(f"[KG_QUERY] Starting get_triples for entity: {entity}, relations: {len(relations)}")
        
        # Augment question with initial entity names (aligning with kgqa_agent)
        augmented_question = self._augment_question(question, session)
        
        resolve_start = time.time()
        entity_resolved = self._resolve_and_register_entity(entity, session)
        resolve_time = time.time() - resolve_start
        if resolve_time > 0.1:
            self._logger.info(f"[TIMING] _handle_get_triples: entity resolution took {resolve_time:.3f}s: {entity} -> {entity_resolved}")
        if not entity_resolved:
            return self._entity_error(entity, session, sample_id)

        validate_start = time.time()
        if session.seen_relations:
            for rel in relations:
                norm_rel = normalize_relation(rel)
                if norm_rel not in session.seen_relations:
                    return self._relation_choice_error(rel, session, sample_id)
        validate_time = time.time() - validate_start
        if validate_time > 0.01:
            self._logger.info(f"[TIMING] _handle_get_triples: relation validation took {validate_time:.3f}s")

        try:
            query_start = time.time()
            self._logger.info(f"[KG_QUERY] Calling _client.get_triples for {entity_resolved} with {len(relations)} relations...")
            result = self._client.get_triples(
                entity_resolved,
                relations,
                limit_per_relation=5,
                question=augmented_question,  # Use augmented_question (aligning with kgqa_agent)
                return_with_cvt_info=True,
            )
            query_time = time.time() - query_start
            if query_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_triples: _client.get_triples took {query_time:.3f}s")
            parse_start = time.time()
            if isinstance(result, dict):
                triples = result.get("triples", [])
                cvt_info = result.get("cvt_info")
            else:
                triples = result
                cvt_info = None
            parse_time = time.time() - parse_start
            if parse_time > 0.01:
                self._logger.info(f"[TIMING] _handle_get_triples: result parsing took {parse_time:.3f}s")
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

        format_start = time.time()
        if not triples:
            formatted = "No triples found."
        else:
            formatted = "\n".join(
                f"[{triple.get('head')}, {triple.get('relation')}, {triple.get('tail')}]"
                for triple in triples
            )
            register_start = time.time()
            self._register_entities(triples, session)
            register_time = time.time() - register_start
            if register_time > 0.01:
                self._logger.info(f"[TIMING] _handle_get_triples: _register_entities took {register_time:.3f}s")
        format_time = time.time() - format_start
        if format_time > 0.1:
            self._logger.info(f"[TIMING] _handle_get_triples: formatting triples took {format_time:.3f}s")

        flatten_start = time.time()
        flatten_relations = self._client.get_pending_flatten_relations(entity_resolved)
        flatten_time = time.time() - flatten_start
        if flatten_time > 0.1:
            self._logger.info(f"[TIMING] _handle_get_triples: get_pending_flatten_relations took {flatten_time:.3f}s")
        
        cvt_start = time.time()
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
        cvt_time = time.time() - cvt_start
        if cvt_time > 0.1:
            self._logger.info(f"[TIMING] _handle_get_triples: CVT info formatting took {cvt_time:.3f}s")

        if flatten_relations:
            rank_start = time.time()
            flatten_rel_dicts = [{"relation": rel} for rel in flatten_relations]
            other_relations = [{"relation": rel} for rel in relations[1:]]
            combined_relations = other_relations + flatten_rel_dicts
            if (
                augmented_question
                and combined_relations
                and hasattr(self._client, "rank_by_similarity")
            ):
                try:
                    rank_sim_start = time.time()
                    # Use augmented_question (question + initial entity names) for BM25 ranking,
                    # aligning with kgqa_agent behavior
                    combined_relations = self._client.rank_by_similarity(
                        combined_relations, augmented_question, "relation"
                    )
                    rank_sim_time = time.time() - rank_sim_start
                    if rank_sim_time > 0.1:
                        self._logger.info(f"[TIMING] _handle_get_triples: rank_by_similarity took {rank_sim_time:.3f}s")
                except Exception:
                    pass
            # Align with kgqa_agent: only filter with LLM if len(combined_relations) > 10
            if len(combined_relations) > 10:
                shuffle_start = time.time()
                random.shuffle(combined_relations)
                shuffle_time = time.time() - shuffle_start
                if shuffle_time > 0.01:
                    self._logger.info(f"[TIMING] _handle_get_triples: shuffle took {shuffle_time:.3f}s")
                filter_start = time.time()
                self._logger.info(f"[KG_FILTER] Applying LLM filter to {len(combined_relations)} combined relations (entity: {entity})")
                # Note: use_flatten_prompt=False aligns with kgqa_agent default behavior
                # (kgqa_agent doesn't pass use_flatten_prompt parameter, so defaults to False)
                combined_relations = self._filter_relations_with_llm(
                    combined_relations, question, entity, use_flatten_prompt=False
                )
                filter_time = time.time() - filter_start
                if filter_time > 0.1:
                    self._logger.info(f"[TIMING] _handle_get_triples: LLM filtering took {filter_time:.3f}s")
                self._logger.info(f"[KG_FILTER] After filtering: {len(combined_relations)} combined relations remaining")
            rank_time = time.time() - rank_start
            if rank_time > 0.1:
                self._logger.info(f"[TIMING] _handle_get_triples: flatten relations processing took {rank_time:.3f}s")
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
        
        total_time = time.time() - total_start
        if total_time > 1.0:
            self._logger.info(f"[TIMING] _handle_get_triples: TOTAL took {total_time:.3f}s")
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
        """Filter relations via LLM to mimic kgqa_agent behaviour.
        
        Note: Filter model is always configured (defaults to "gpt-4o-mini" if not provided),
        aligning with kgqa_agent behavior.
        """
        import time
        filter_start = time.time()
        
        # Log filter start with detailed information
        self._logger.info(
            f"[KG_FILTER] Starting filter: entity='{entity_name}', "
            f"input_relations={len(relations)}, use_flatten_prompt={use_flatten_prompt}"
        )
        
        if not relations:
            self._logger.info("[KG_FILTER] No relations to filter, returning empty list")
            return relations

        # Keep only entries that actually contain relation names
        relation_entries = [rel for rel in relations if rel.get("relation")]
        if len(relation_entries) < len(relations):
            self._logger.warning(
                f"[KG_FILTER] Filtered out {len(relations) - len(relation_entries)} relations "
                f"without 'relation' field (input={len(relations)}, valid={len(relation_entries)})"
            )
        
        # Always apply LLM filter regardless of relation count (aligning with kgqa_agent behavior)
        
        ensure_start = time.time()
        self._ensure_filter_client()
        ensure_time = time.time() - ensure_start
        if ensure_time > 0.01:
            self._logger.info(f"[TIMING] _filter_relations_with_llm: _ensure_filter_client took {ensure_time:.3f}s")
        if self._filter_client is None:
            # Failed to build client; fall back immediately
            self._logger.warning(
                f"[KG_FILTER] Filter client not available, returning {len(relation_entries)} relations without filtering"
            )
            return relation_entries

        # Shuffle before building prompt to avoid positional bias
        prep_start = time.time()
        shuffled = relation_entries[:]
        random.shuffle(shuffled)
        
        # Log relation names being filtered (truncate if too long)
        relation_names = [rel['relation'] for rel in shuffled]
        if len(relation_names) <= 20:
            self._logger.info(f"[KG_FILTER] Filtering {len(relation_names)} relations: {relation_names}")
        else:
            self._logger.info(
                f"[KG_FILTER] Filtering {len(relation_names)} relations: "
                f"{relation_names[:10]} ... {relation_names[-10:]}"
            )

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
        prep_time = time.time() - prep_start
        if prep_time > 0.01:
            self._logger.info(f"[TIMING] _filter_relations_with_llm: prompt preparation took {prep_time:.3f}s")
        
        # Log prompt info
        self._logger.info(
            f"[KG_FILTER] Prompt prepared: length={len(prompt)}, "
            f"template_type={'flatten' if use_flatten_prompt else 'standard'}"
        )

        max_retries = 3
        last_exception: Optional[Exception] = None
        json_pattern = re.compile(r"\[.*?\]", re.DOTALL)

        for attempt in range(max_retries):
            self._filter_total_calls += 1
            self._logger.info(
                f"[KG_FILTER] LLM call attempt {attempt + 1}/{max_retries} "
                f"(total_calls={self._filter_total_calls})"
            )
            try:
                llm_start = time.time()
                response = self._filter_client.generate(
                    prompt,
                    temperature=self._relation_filter_temperature,
                    max_tokens=self._relation_filter_max_tokens,
                )
                llm_time = time.time() - llm_start
                
                # Log LLM response info
                response_preview = (response or "")[:200] if response else "None"
                self._logger.info(
                    f"[KG_FILTER] LLM response received (attempt {attempt+1}): "
                    f"time={llm_time:.3f}s, length={len(response) if response else 0}, "
                    f"preview='{response_preview}...'"
                )
                
                if llm_time > 0.1:
                    self._logger.info(f"[TIMING] _filter_relations_with_llm: LLM generate (attempt {attempt+1}) took {llm_time:.3f}s")
                
                parse_start = time.time()
                json_match = json_pattern.search(response or "")
                if not json_match:
                    self._filter_parse_fail_count += 1
                    self._logger.warning(
                        f"[KG_FILTER] Failed to find JSON array in response (attempt {attempt+1}): "
                        f"response_preview='{response_preview}...'"
                    )
                    break

                parsed = json.loads(json_match.group(0))
                if not isinstance(parsed, list):
                    self._filter_parse_fail_count += 1
                    self._logger.warning(
                        f"[KG_FILTER] Parsed JSON is not a list (attempt {attempt+1}): "
                        f"type={type(parsed)}, value={parsed}"
                    )
                    break

                rel_map = {rel["relation"]: rel for rel in relation_entries}
                ordered_filtered = [
                    rel_map[name]
                    for name in parsed
                    if isinstance(name, str) and name in rel_map
                ]
                
                # Log filtering results
                filtered_out = len(relation_entries) - len(ordered_filtered)
                parsed_but_not_found = [name for name in parsed if not (isinstance(name, str) and name in rel_map)]
                if parsed_but_not_found:
                    self._logger.warning(
                        f"[KG_FILTER] LLM selected relations not in input (attempt {attempt+1}): "
                        f"{parsed_but_not_found}"
                    )
                
                parse_time = time.time() - parse_start
                if parse_time > 0.01:
                    self._logger.info(f"[TIMING] _filter_relations_with_llm: response parsing took {parse_time:.3f}s")
                
                if ordered_filtered:
                    self._filter_success_count += 1
                    self._maybe_log_filter_stats()
                    total_time = time.time() - filter_start
                    
                    # Log successful filter result
                    filtered_relation_names = [rel['relation'] for rel in ordered_filtered]
                    if len(filtered_relation_names) <= 20:
                        self._logger.info(
                            f"[KG_FILTER] Filter SUCCESS (attempt {attempt+1}): "
                            f"kept {len(ordered_filtered)}/{len(relation_entries)} relations: {filtered_relation_names}"
                        )
                    else:
                        self._logger.info(
                            f"[KG_FILTER] Filter SUCCESS (attempt {attempt+1}): "
                            f"kept {len(ordered_filtered)}/{len(relation_entries)} relations: "
                            f"{filtered_relation_names[:10]} ... {filtered_relation_names[-10:]}"
                        )
                    
                    if total_time > 0.1:
                        self._logger.info(f"[TIMING] _filter_relations_with_llm: TOTAL took {total_time:.3f}s, filtered {len(ordered_filtered)}/{len(relation_entries)} relations")
                    return ordered_filtered

                # If parsed successfully but no overlap, treat as fallback
                self._filter_fallback_count += 1
                self._logger.warning(
                    f"[KG_FILTER] LLM returned valid JSON but no overlap with input relations "
                    f"(attempt {attempt+1}): parsed={parsed}, input_count={len(relation_entries)}"
                )
                break
            except Exception as exc:  # pragma: no cover - network/LLM errors
                last_exception = exc
                if attempt + 1 < max_retries:
                    retry_delay = min(4.0, 1.5 ** attempt)
                    self._logger.warning(
                        f"[KG_FILTER] LLM call failed (attempt {attempt+1}/{max_retries}): {exc}, "
                        f"retrying in {retry_delay:.2f}s"
                    )
                    time.sleep(retry_delay)
                    continue
                self._filter_fallback_count += 1
                self._logger.error(
                    f"[KG_FILTER] LLM filter failed after {attempt + 1} attempts: {exc}",
                    exc_info=True
                )

        self._maybe_log_filter_stats()
        
        # Log fallback result
        self._logger.warning(
            f"[KG_FILTER] Filter FALLBACK: returning all {len(relation_entries)} input relations "
            f"(entity='{entity_name}')"
        )
        return relation_entries

    def _maybe_log_filter_stats(self) -> None:
        now = time.time()
        # Log at most once per minute to avoid spam
        if now - self._filter_last_log_time < 60:
            return
        self._filter_last_log_time = now
        # Filter model is always configured
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
        import time
        resolve_start = time.time()
        
        if not entity:
            return None
        
        # Reject strings that look like error messages to prevent them from being used as entity names
        validate_start = time.time()
        error_keywords = [
            "Could not parse",
            "Invalid query",
            "Query parsing failed",
            "Invalid entity",
            "Error:",
        ]
        if any(keyword in entity for keyword in error_keywords):
            self._logger.warning(
                f"Rejected invalid entity name (looks like error message): {entity[:100]}"
            )
            return None
        validate_time = time.time() - validate_start
        if validate_time > 0.01:
            self._logger.info(f"[TIMING] _resolve_and_register_entity: validation took {validate_time:.3f}s")
        
        check_start = time.time()
        if entity.startswith(("m.", "g.", "en.")):
            session.entity_registry[entity] = entity
            return entity

        key = entity.lower()
        if key in session.entity_registry:
            cached = session.entity_registry[key]
            check_time = time.time() - check_start
            if check_time > 0.01:
                self._logger.info(f"[TIMING] _resolve_and_register_entity: cache lookup took {check_time:.3f}s")
            return cached
        check_time = time.time() - check_start
        if check_time > 0.01:
            self._logger.info(f"[TIMING] _resolve_and_register_entity: cache check took {check_time:.3f}s")

        resolve_entity_start = time.time()
        resolved = self._client._resolve_entity(entity)
        resolve_entity_time = time.time() - resolve_entity_start
        if resolve_entity_time > 0.1:
            self._logger.info(f"[TIMING] _resolve_and_register_entity: _client._resolve_entity took {resolve_entity_time:.3f}s for '{entity}'")
        
        if resolved:
            # Validate resolved entity_id format (should be MID)
            if not resolved.startswith(("m.", "g.", "en.")):
                self._logger.warning(
                    f"Resolved entity_id has invalid format: {resolved} (expected MID format)"
                )
                return None
            session.entity_registry[key] = resolved
            session.entity_registry[resolved] = entity
        
        total_time = time.time() - resolve_start
        if total_time > 0.1:
            self._logger.info(f"[TIMING] _resolve_and_register_entity: TOTAL took {total_time:.3f}s for '{entity}' -> {resolved}")
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


