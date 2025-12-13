"""
LLM prompt for per-step planning and top-k candidate filtering (concise, two-line output).

Output format (exactly two lines):
1) <plan>one sentence</plan>
2) <keep>[up to k keys]</keep>

Keys rule: if a candidate has id use id, else use name. Keys must be taken verbatim from candidates. Always include the golden item. Never exceed k.
"""

TRAJECTORY_PLAN_AND_FILTER_PROMPT = """
You are an expert at pruning knowledge graph candidates per reasoning step.

Given:
- raw_question: natural language question
- name_path: the golden reasoning path (entity -> relation -> entity -> ...)
- step: JSON with fields { step_index, query_type, current, args }
- candidates: up to 20 items.
- golden: the golden item { id?: string, name?: string } guaranteed to be in candidates
- k: max number to keep (default 5)

Task:
1) Plan: Write ONE short sentence that focuses on moving from the current item to the next:
  - If query_type contains "relations": start from the current entity and state how to find the next relation.
  - If query_type contains "entities": start from the current relation (and entity) and state how to find the next entity.
2) Keep: Select up to k items most relevant to the question and sub-goal (finding the next hop). MUST include golden items! Choose only from candidates and copy them EXACTLY.

Selection tips (next-hop oriented):
- Match the question intent and stay consistent with name_path.
- For relation queries: from the current entity, keep relations that advance toward the next entity/hop.
- For entity queries: under the current relation, keep entities that are the plausible next hop from the current entity.

STRICT output (two lines only):
<plan><one sentence></plan>
<keep>["key1", "key2", ...]</keep>

Complete example (from dataset):
raw_question: "Who authored the book about the Hindu belief in reincarnation?"
name_path: "Hinduism -> religion.religion.beliefs -> Reincarnation -> book.book_subject.works -> Reincarnation and Biology -> book.written_work.author -> Ian Stevenson"
step: {"step_index": 0, "query_type": "get_relations", "current": "Hinduism", "args": {"entity": "m.03j6c", "top_k": 20}}
candidates: [
          "book.book_subject.works",
          "religion.religion.types_of_places_of_worship",
          "religion.religion.holidays",
          "religion.religion.beliefs",
          "religion.religion.is_part_of",
          "religion.religion.organizations",
          "religion.religion.deities",
          "religion.religion.founding_figures",
          "religion.religion.practices",
          "religion.religion.places_of_worship",
          "religion.religion.texts"
        ]
golden: ["religion.religion.beliefs"]
k: 5

Desired output:
<plan>From Hinduism, I will query for its relations and select a relation that reveals its beliefs to reach Reincarnation.</plan>
<keep>["religion.religion.beliefs", "religion.religion.practices", "religion.religion.deities", "book.book_subject.works", "religion.religion.texts"]</keep>
"""