"""
Utilities for loading and grouping the Turkish Extended NER tagset.
"""

from collections import defaultdict
from typing import Dict, List, Set


DEFAULT_TAGSET_PATH = "data/ENER-tagset.tsv"


def load_tagset(path: str = DEFAULT_TAGSET_PATH) -> List[str]:
    """
    Load the ENER tagset from a CSV-like file.

    Expected format (one tag per line, possibly with trailing comma):
        Named Entity tags,Named Entity annotation   <-- header (ignored)
        ACT,
        AGE,
        FAC_AIRPORT,
        PRO_LANGUAGE,
        ...

    Returns a sorted list of unique tag strings.
    """
    tags: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip header line
            if line_num == 0 and "Named Entity" in line:
                continue

            # Split by comma, take the first non-empty chunk
            first = line.split(",")[0].strip()
            if first:
                tags.append(first)

    # Remove duplicates, keep sorted for reproducibility
    unique_tags = sorted(set(tags))
    return unique_tags


def group_tagset_by_prefix(tags: List[str]) -> Dict[str, List[str]]:
    """
    Group tags by their prefix before the first underscore.

    Examples:
        FAC_AIRPORT  -> group 'FAC'
        LOC_CITY     -> group 'LOC'
        PRO_LANGUAGE -> group 'PRO'
        PER          -> group 'BASE'
        ORG          -> group 'BASE'

    Returns a dict: prefix -> list of tags
    """
    groups: Dict[str, List[str]] = defaultdict(list)

    for tag in tags:
        if "_" in tag:
            prefix = tag.split("_", 1)[0]
        else:
            prefix = "BASE"  # no prefix, core type
        groups[prefix].append(tag)

    # Sort tags inside each group
    for prefix in groups:
        groups[prefix] = sorted(groups[prefix])

    return dict(groups)


def compare_tagset_with_corpus(
    tagset: List[str],
    corpus_entity_types: Set[str],
) -> Dict[str, List[str]]:
    """
    Compare the tagset with the entity types that actually appear in the corpus.

    Parameters
    ----------
    tagset : list of tag strings (e.g. ['PERSON', 'LOC_CITY', 'PRO_LANGUAGE', ...])
    corpus_entity_types : set of entity type strings extracted from the corpus
        (e.g. keys of type_counter in statistics.py)

    Returns
    -------
    A dictionary with:
        - 'unused_in_corpus': tags defined in the tagset but never seen in the corpus
        - 'unknown_in_tagset': types seen in the corpus but not defined in the tagset
    """
    tagset_set = set(tagset)

    unused = sorted(tagset_set - corpus_entity_types)
    unknown = sorted(corpus_entity_types - tagset_set)

    return {
        "unused_in_corpus": unused,
        "unknown_in_tagset": unknown,
    }
