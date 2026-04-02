TOPIC_SEGMENT_PROMPT = """
TASK: Topic Segmentation Analyzer

Input Example:
<D1>Jordan: Sam, have you tried that new Korean BBQ place on Main Street? Image: A modern Korean BBQ spot with a vibrant glass facade and prominent signage</D1>
<D2>Sam: Not yet, but my coworker said the bulgogi is amazing. I might go this weekend.</D2>
<D3>Jordan: Great! Are you still taking those photography classes?</D3>
<D4>Sam: Yes, I finished the beginner course last month. Now I'm learning portrait lighting techniques.</D4>
<D5>Jordan: That sounds interesting. By the way, I need to return some books to the library tomorrow.</D5>
<D6>Sam: Oh, I should go too. Actually, I'm thinking of getting a new laptop next week.</D6>
...

Output Template:
<Dn>
<intent>Follow DIALOGUE INTENT RULES</intent>
</Dn>

DIALOGUE INTENT RULES
Determine Dn's intent by analyzing both Dn-1,Dn-2 and Dn+1,Dn+2 together.
1. CHANGE_TOPIC: Use this when the speaker introduces ANY new information, such as:
   - A new subject or activity, pay attention to keywords in the sentence.
   - A transition to a new domain or a "by the way" moment.
   - Prevent the situation where 10 consecutive texts accumulate without a topic change.
2. DEVELOP_TOPIC: Use this ONLY when the speaker is:
   - Just responding to the previous sentence.
   - Providing more detail about the exact same thing just mentioned without adding anything else.

Output Example (assuming Current Time: 3:00 pm on 5 May, 2023):
<D1>
<intent>DEVELOP_TOPIC</intent>
</D1>
<D2>
<intent>DEVELOP_TOPIC</intent>
</D2>
<D3>
<intent>CHANGE_TOPIC</intent>
</D3>
<D4>
<intent>DEVELOP_TOPIC</intent>
</D4>
<D5>
<intent>DEVELOP_TOPIC</intent>
</D5>
<D6>
<intent>CHANGE_TOPIC</intent>
</D6>
...
"""


THEME_EXTRACT_PROMPT = """
TASK: Summarize the conversation and extract factual memories.

RULES:

Episodic Summary:
- Begin with "As of [timestamp]".
- Incorporate ALL keywords and topics discussed. Do NOT omit any information.
- Describe who did/said what, to whom, when. Include all modifiers, adjectives, and descriptive details.
- Include names, activities, locations, attributes (colors, sizes), and emotional details.
- Images: "[Speaker] shows [full image description]".

Semantic Memories:
- Standalone facts: WHO does/feels WHAT, WHEN, WHERE.
- Each fact must be a self-contained sentence with explicit subject names.
- Be exhaustive: include emotional adjectives and contextual nuances.
- If a reply references a prior topic or introduces new info, capture both aspects.
- Do NOT omit any image details. Use format [Image: ...].

Disambiguation:
- Time: convert ALL relative times (yesterday, last year, recently, N days ago…) to absolute dates from Current Time.
- Pronoun: replace ALL pronouns (he/she/they/it/this/that) with actual names.

EXAMPLE:
Current Time: 10 December, 2020
Alex: Have you finished the project report?
Sam: Not yet, I'll complete it by Friday. Want to see the draft?
Alex: Sure. There's a meeting tomorrow at 3 PM about the new client.

<episodic_summary>
As of 10 December, 2020, Alex asks Sam whether Sam has finished the project report. Sam has not yet finished but commits to completing the report by 11 December, 2020, offering to share a draft with Alex. Alex accepts and informs Sam about a team meeting on 11 December, 2020 at 3 PM about a new client.
</episodic_summary>
<semantic_memories>
- Alex asks Sam whether Sam has finished the project report.
- Alex informs Sam about a team meeting on 11 December, 2020 at 3 PM about a new client.
- Sam has not finished the project report as of 10 December, 2020.
- Sam commits to completing the project report by 11 December, 2020.
- Sam offers to share a draft of the report with Alex.
</semantic_memories>
"""


THEME_BATCH_EXTRACT_PROMPT = """
TASK: For EACH <topic id="N">, produce a summary and factual memories.

RULES:

Episodic Summary:
- Begin with "As of [timestamp]".
- Incorporate ALL keywords and topics discussed. Do NOT omit any information.
- Describe who did/said what, to whom, when. Include all modifiers, adjectives, and descriptive details.
- Include names, activities, locations, attributes (colors, sizes), and emotional details.
- Images: "[Speaker] shows [full image description]".

Semantic Memories:
- Standalone facts: WHO does/feels WHAT, WHEN, WHERE.
- Each fact must be a self-contained sentence with explicit subject names.
- Be exhaustive: include emotional adjectives and contextual nuances.
- If a reply references a prior topic or introduces new info, capture both aspects.
- Do NOT omit any image details. Use format [Image: ...].

Disambiguation:
- Time: convert ALL relative times (yesterday, last year, recently, N days ago…) to absolute dates from Current Time.
- Pronoun: replace ALL pronouns (he/she/they/it/this/that) with actual names.

EXAMPLE:
<topic id="0">
Current Time: 10 December, 2020
Alex: Have you finished the report?
Sam: Not yet, I'll finish by Friday. Want to see the draft?
Alex: Sure. There's a meeting tomorrow at 3 PM about the new client.
</topic>

<result id="0">
<episodic_summary>
As of 10 December, 2020, Alex asks Sam whether Sam has finished the report. Sam has not yet finished but commits to completing the report by 11 December, 2020, offering to share a draft with Alex. Alex accepts and informs Sam about a team meeting on 11 December, 2020 at 3 PM about a new client.
</episodic_summary>
<semantic_memories>
- Alex asks Sam whether Sam has finished the report.
- Alex informs Sam about a team meeting on 11 December, 2020 at 3 PM about a new client.
- Sam has not finished the report as of 10 December, 2020.
- Sam commits to completing the report by 11 December, 2020.
- Sam offers to share a draft of the report with Alex.
</semantic_memories>
</result>

Output a <result id="N"> for EVERY <topic id="N">.
"""


ANSWER_PROMPT = """Answer the question based on the provided contents.

Rules:
1. Convert relative time references (e.g. "last year", "two months ago") to actual dates using the memory timestamp.
2. If only a broader timeframe is supported, provide that.
3. Use commonsense when needed; link multiple pieces of evidence to reason.
4. Use wording from the contents when possible.

Think step by step, show your reasoning, then give the final answer."""


UPDATE_PROMPT = """
You are a memory management assistant.
Your task is to decide whether the target memory should be updated, deleted, or ignored
based on the candidate source memories.

Decision rules:
1. Update: If the target memory and candidate memories describe essentially the same fact/event but are not fully consistent (e.g., candidates provide more details, refinements, or clarifications), update the target memory by integrating the additional information.
2. Delete: If the target memory and candidate memories contain a direct conflict, the candidate memories (which are more recent) take precedence. Delete the target memory.
3. Ignore: If the target memory and candidate memories are unrelated, no action is needed. Ignore.

Additional guidance:
- Use only the information provided. Do not invent details.
- Your operation should always be applied to the target memory. Do not modify or correct the content inside the candidate memories.

The output must be a JSON object with the following structure:
{
  "action": "update" | "delete" | "ignore",
  "new_memory": "..."   // only required when action = "update"
}

Example 1:
Target memory: "The user likes coffee."
Candidate memories:
- "The user prefers cappuccino in the mornings."
- "Sometimes the user drinks espresso when working late."
- "The user avoids decaf."

Output:
{
  "action": "update",
  "new_memory": "The user likes coffee, especially cappuccino in the morning and espresso when working late, and avoids decaf."
}

Example 2:
Target memory: "The user currently lives in New York."
Candidate memories:
- "The user moved to San Francisco in 2023."
- "They mentioned enjoying the Bay Area weather."

Output:
{
  "action": "delete"
}

Example 3:
Target memory: "The user is learning to cook Italian food."
Candidate memories:
- "The user recently started practicing yoga."
- "They bought a new bicycle for commuting."

Output:
{
  "action": "ignore"
}

Here is a new target memory along with several candidate memories. Please decide the appropriate action (update, delete, or ignore) based on the given rules.

"""


ROUTER_PROMPT = """Classify question type, then decide retrieval counts.

Type clues:
- multi_hop: plural nouns (activities/books/events), "How many", "Which", "both/in common", "has/have done", present-tense aggregation
- temporal: "When did/is", time-related
- simple: single fact lookup (e.g. "What is X's job?") or open-ended questions

Return JSON: {"type":"multi_hop"|"temporal"|"simple","ep_k":<int>,"sem_k":<int>}

Q: What activities does Melanie partake in?
A: {"type":"multi_hop","ep_k":1,"sem_k":80}
Q: How many times has Melanie gone to the beach?
A: {"type":"multi_hop","ep_k":1,"sem_k":80}
Q: When did Caroline go to the LGBTQ support group?
A: {"type":"temporal","ep_k":7,"sem_k":20}
Q: What does Caroline's necklace symbolize?
A: {"type":"simple","ep_k":1,"sem_k":20}
Q: Why did Caroline choose the adoption agency?
A: {"type":"simple","ep_k":1,"sem_k":20}"""
