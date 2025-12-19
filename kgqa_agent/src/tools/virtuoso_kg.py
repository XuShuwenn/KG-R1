from __future__ import annotations
import time
import os
from typing import Optional

from SPARQLWrapper import SPARQLWrapper, JSON


def _setup_sparql_no_proxy():
    """Setup environment to bypass proxy for SPARQL requests."""
    # Add localhost and 127.0.0.1 to no_proxy to bypass proxy for local SPARQL endpoints
    no_proxy = os.environ.get('no_proxy', '')
    no_proxy_list = [x.strip() for x in no_proxy.split(',') if x.strip()]
    for host in ['localhost', '127.0.0.1', '0.0.0.0']:
        if host not in no_proxy_list:
            no_proxy_list.append(host)
    os.environ['no_proxy'] = ','.join(no_proxy_list)
    # Also set NO_PROXY (uppercase) for compatibility
    os.environ['NO_PROXY'] = os.environ['no_proxy']


class VirtuosoKG:
    """
    Lightweight wrapper for connecting to a Virtuoso SPARQL endpoint and
    executing queries.

    Preserves behavior: timeout configuration, retry with exponential backoff,
    and a simple name cache for entity URIs.
    """
    def __init__(self, endpoint_url: str = "http://localhost:18890/sparql", timeout_s: int = 15):
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s
        self.sparql = None
        self._new_client()
        self.name_cache: dict[str, str] = {}

    def _new_client(self):
        # Setup no_proxy for local SPARQL endpoints
        _setup_sparql_no_proxy()
        self.sparql = SPARQLWrapper(self.endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(self.timeout_s)

    def execute_query(self, query: str, verbose: bool = False, retries: int = 3, backoff_base: float = 0.75, reset_on_fail: bool = True):
        last_err = None
        for attempt in range(1, max(1, retries) + 1):
            start_time = time.time()
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                end_time = time.time()
                if verbose:
                    print(f"    Query succeeded, elapsed: {end_time - start_time:.2f} s")
                return results.get("results", {}).get("bindings", [])
            except Exception as e:
                last_err = e
                end_time = time.time()
                if verbose:
                    print(f"    Query failed (elapsed: {end_time - start_time:.2f} s): {e} (attempt {attempt}/{retries})")
                if reset_on_fail:
                    self._new_client()
                if attempt < retries:
                    time.sleep(backoff_base * (2 ** (attempt - 1)))
        return []

    def get_entity_name(self, entity_uri: Optional[str], graph_uri: Optional[str] = None):
        if not entity_uri:
            return "N/A"
        if entity_uri in self.name_cache:
            return self.name_cache[entity_uri]
        inner = f"""
          OPTIONAL {{ <{entity_uri}> ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en),'en')) }}
          OPTIONAL {{ <{entity_uri}> ns:type.object.name ?name_zh . FILTER(LANGMATCHES(LANG(?name_zh),'zh')) }}
          OPTIONAL {{ <{entity_uri}> ns:type.object.alias ?alias_en . FILTER(LANGMATCHES(LANG(?alias_en),'en')) }}
          OPTIONAL {{ <{entity_uri}> ns:common.topic.description ?desc_en . FILTER(LANGMATCHES(LANG(?desc_en),'en')) }}
          OPTIONAL {{ <{entity_uri}> ns:type.object.name ?name_any . }}
        """
        where = f"GRAPH <{graph_uri}> {{ {inner} }}" if graph_uri else inner
        query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT (COALESCE(?name_en, ?name_zh, ?alias_en, ?desc_en, ?name_any) AS ?name)
        WHERE {{ {where} }}
        LIMIT 1
        """
        bindings = self.execute_query(query, verbose=False)
        if bindings:
            first = bindings[0]
            if "name" in first and first["name"].get("value"):
                val = first["name"]["value"]
                self.name_cache[entity_uri] = val
                return val
        tail = entity_uri.rsplit('#', 1)[-1]
        tail = tail.rsplit('/', 1)[-1]
        val = tail or "N/A"
        self.name_cache[entity_uri] = val
        return val
