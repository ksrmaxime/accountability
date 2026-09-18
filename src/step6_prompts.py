# src/step6_prompts.py
"""
STEP 6 — admin response: does the target respond to the criticism directed
at it?

Ported from the legacy run7 stage with the SYSTEM_PROMPT wording untouched.
The USER_TEMPLATE's final two instructions were reordered (justification
now comes before the YES/NO answer instead of after) for consistency with
steps 3, 4a, 4b and 5, which all now use the same "justification first"
pattern — the sentence-level wording of both instructions is otherwise
unchanged. The other real change is how the `{source}` slot is filled: the
legacy pipeline had a free-text
`critic_answer_final` column (an open-ended "who criticizes?" answer,
arbitrated/validated through run4arbitre + run5). That column does not exist
in this pipeline — it has been replaced by step 4a's binary
SOURCED/JOURNALIST call and step 4b's 6-way source_category. `_source_phrase`
below turns those two columns back into a short natural-language phrase to
slot into the original sentence ("... is criticised by {source}."), so the
prompt text itself is identical to before.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
Your task is to determine whether, in a newspaper article, the entity being
criticised responds to the criticism/reproach directed at them.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticised by {source}.

ARTICLE:
{article_text}

Your task is to determine whether, in the article, "{keyword}" gives a response \
to the reproach addressed to them.

First, in one sentence, briefly justify your answer based solely on the article.
Then, on a new line, answer with exactly one word: YES or NO.\
"""

# step 4b source_category -> natural-language phrase, in the register of the
# original free-text critic_answer_final values it replaces.
_CATEGORY_PHRASES = {
    "Interest Group": "an interest group",
    "Civil Servant": "a civil servant",
    "General Public": "a member of the general public",
    "Politician": "a politician",
    "Administrative Unit of the State": "another administrative unit of the state",
    "Other": "an external actor",
}


def _source_phrase(row: pd.Series) -> str:
    attribution = row.get("source_attribution", pd.NA)
    attribution = "" if pd.isna(attribution) else str(attribution).strip().upper()

    if attribution == "JOURNALIST":
        return "the journalist"

    if attribution == "SOURCED":
        category = row.get("source_category", pd.NA)
        category = "" if pd.isna(category) else str(category).strip()
        return _CATEGORY_PHRASES.get(category, "an unspecified source")

    # source_attribution missing/unparsed (should not normally happen since
    # step 4a covers every keyword_answer == "YES" row) -> fall back exactly
    # like the legacy code did when critic_answer_final was NA.
    return "an unspecified source"


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    source = _source_phrase(row)
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text, source=source)
