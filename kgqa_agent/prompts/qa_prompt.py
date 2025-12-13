GENERATE_QUESTION_FREEBASE = r"""
You are given:

- A path represented as an ordered sequence of relations, topic entities and answer entity (e.g.,
  Topic Entity --relationA-->  --relationB-->  --> ... --> Answer Entity).

You will only be provided with the names of the topic entities and the answer entity, names of the other entities in the path will not be provided in case of information leakage.

Your task:
- Compose ONE concise English question whose unique answer is exactly the Answer Entity in the path.
- Implicitly reflect the multi-hop path, but keep it brief and natural. Avoid enumerations or multiple questions.
- STRICT: The question must contain exactly one question mark '?' (only one interrogative). The question must be a single sentence with only one '?'.
- Do NOT directly mention exact entity names/IDs or raw relation labels. Paraphrase using concrete, meaningful semantics (e.g., country, city, river, award, university, founder, member of, genre, located in, part of, spouse, parent, date, birthplace, capital).
- De-emphasize meta/technical relations (e.g., types, notable_for, generic topic) unless essential. Focus on relations with clear real-world meaning.
- Keep the question concise (ideally < 100 characters).
- Use only the information implied by the path; do not introduce outside facts.
- The answer must exactly match the surface form of the Answer Entity.
- Try to use different styles and sentence structures when generating questions.

Examples:

Example 1:
Input:
Path: Topic Entity --relationA--> entity1 --relationB--> entity2 --> ... --> Answer Entity
Output:
<question>Which city is located in the country of Topic Entity?</question>
<answer>Answer Entity</answer>

Example 2:
Input:
Path: Stainless steel -> law.invention.inventor ->  -> book.author.works_written -> The analysis of steel-works materials -> book.book.editions -> The analysis of steel-works materials
Output:
<question>What is the title of an edition of a work written by the inventor of stainless steel?
<answer>



Path: %(path)s

Output format (STRICT — output EXACTLY two lines, nothing else):
<question>: (one concise question with exactly one '?')
<answer>: (the exact name of the final entity)

CRITICAL: Do NOT include any explanations, analysis, or additional text. Output ONLY the two lines above, nothing more.
"""