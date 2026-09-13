"""
Hierarchical Community GraphRAG Summarizer Module.

Provides global summarization, centrality-based token budgeting,
multi-tier LLM unwrapping, thread-safe SHA-256 caching, and bottom-up
hierarchical synthesis for community reports.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.logging import get_logger
from .centrality_calculator import CentralityCalculator
from .community_hierarchy import (
    CommunityHierarchy,
    HierarchicalCommunity,
    compute_community_hash,
)

logger = get_logger("community_summarizer")


def estimate_tokens(
    text: str,
    custom_counter: Optional[Callable[[str], int]] = None,
) -> int:
    """
    Estimate token count for a text string.

    Args:
        text: Input string to measure.
        custom_counter: Optional callable accepting string and returning int.

    Returns:
        Estimated number of tokens (0 if text is empty).
    """
    if not text:
        return 0
    if custom_counter is not None and callable(custom_counter):
        try:
            return max(0, int(custom_counter(text)))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4.0))


@dataclass
class CommunityReport:
    """
    Structured summary report for a knowledge graph community.

    Represents an executive-level summary and detailed findings synthesized
    from member entities, internal relationships, and finer child communities.
    """

    community_id: str
    level: int
    title: str
    summary: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    impact_rating: float = 5.0
    rating_explanation: str = ""
    member_entities: List[str] = field(default_factory=list)
    content_hash: str = ""
    sub_communities: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    rank: float = 0.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.community_id = str(self.community_id)
        self.level = int(self.level)
        self.title = str(self.title).strip()
        if not self.title:
            self.title = f"Community {self.community_id}"
        self.summary = str(self.summary).strip()
        self.rating_explanation = str(self.rating_explanation).strip()

        try:
            r_val = float(self.impact_rating)
            if math.isnan(r_val) or math.isinf(r_val):
                r_val = 5.0
        except (ValueError, TypeError):
            r_val = 5.0
        self.impact_rating = max(1.0, min(10.0, r_val))

        try:
            rank_val = float(self.rank)
            if math.isnan(rank_val) or math.isinf(rank_val):
                rank_val = 0.0
        except (ValueError, TypeError):
            rank_val = 0.0
        self.rank = rank_val

        if self.member_entities:
            self.member_entities = sorted(
                set(str(e) for e in self.member_entities)
            )
        else:
            self.member_entities = []

        if self.sub_communities:
            self.sub_communities = sorted(
                set(str(c) for c in self.sub_communities)
            )
        else:
            self.sub_communities = []

        if self.parent_id is not None:
            self.parent_id = str(self.parent_id)

        if self.embedding is not None:
            self.embedding = [float(x) for x in self.embedding]

        if not isinstance(self.findings, list):
            self.findings = []
        else:
            norm_findings = []
            for item in self.findings:
                if isinstance(item, dict):
                    norm_findings.append(
                        {str(k): v for k, v in item.items()}
                    )
                else:
                    norm_findings.append(
                        {"summary": str(item), "explanation": ""}
                    )
            self.findings = norm_findings

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize community report to a dictionary."""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "findings": [
                dict(item) if isinstance(item, dict) else item
                for item in self.findings
            ],
            "impact_rating": self.impact_rating,
            "rating_explanation": self.rating_explanation,
            "member_entities": list(self.member_entities),
            "content_hash": self.content_hash,
            "sub_communities": list(self.sub_communities),
            "parent_id": self.parent_id,
            "rank": self.rank,
            "embedding": (
                list(self.embedding) if self.embedding is not None else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunityReport":
        """Instantiate a community report from a dictionary."""
        raw_parent = data.get("parent_id")
        parent_id = str(raw_parent) if raw_parent is not None else None
        raw_embed = data.get("embedding")
        embedding = (
            [float(x) for x in raw_embed] if raw_embed is not None else None
        )
        findings = [
            dict(item) if isinstance(item, dict) else item
            for item in data.get("findings", [])
        ]
        return cls(
            community_id=str(data.get("community_id", "")),
            level=int(data.get("level", 0)),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            findings=findings,
            impact_rating=float(data.get("impact_rating", 5.0)),
            rating_explanation=str(data.get("rating_explanation", "")),
            member_entities=list(data.get("member_entities", [])),
            content_hash=str(data.get("content_hash", "")),
            sub_communities=list(data.get("sub_communities", [])),
            parent_id=parent_id,
            rank=float(data.get("rank", 0.0)),
            embedding=embedding,
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CommunityReport":
        """Instantiate report from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_markdown(self) -> str:
        """Format report into executive markdown presentation."""
        lines = [
            f"# {self.title}",
            "",
            f"**Community ID:** {self.community_id}  ",
            f"**Level:** {self.level}  ",
            f"**Impact Rating:** {self.impact_rating:.1f}/10  ",
        ]
        if self.rating_explanation:
            lines.append(f"*{self.rating_explanation}*")
        lines.extend([
            "",
            "## Summary",
            "",
            self.summary if self.summary else "No summary provided.",
            "",
            "## Key Findings",
            "",
        ])
        if self.findings:
            for idx, finding in enumerate(self.findings, 1):
                if isinstance(finding, dict):
                    heading = (
                        finding.get("summary")
                        or finding.get("title")
                        or finding.get("finding")
                        or finding.get("name")
                        or finding.get("claim")
                        or f"Finding {idx}"
                    )
                    explanation = (
                        finding.get("explanation")
                        or finding.get("description")
                        or finding.get("detail")
                        or finding.get("evidence")
                        or ""
                    )
                    if explanation:
                        lines.append(f"- **{heading}**: {explanation}")
                    else:
                        lines.append(f"- **{heading}**")
                else:
                    lines.append(f"- {finding}")
        else:
            lines.append("No specific findings reported.")

        if self.member_entities:
            lines.extend([
                "",
                "## Member Entities",
                "",
                ", ".join(self.member_entities),
            ])
        return "\n".join(lines)


class CommunityReportLLMSchema(BaseModel):
    """
    Pydantic schema for structured LLM community report generation.

    Includes resilient field validators to normalize model outputs,
    clamp numeric ratings, and coerce findings into lists of dictionaries.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str = Field(
        default="Community Summary",
        description="Concise theme title for the community.",
    )
    summary: str = Field(
        default="",
        description="Comprehensive summary of entities and dynamics.",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured key findings and supporting context.",
    )
    impact_rating: float = Field(
        default=5.0,
        description="Impact severity rating clamped between 1.0 and 10.0.",
    )
    rating_explanation: str = Field(
        default="",
        description="Rationale justifying the assigned impact rating.",
    )

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, v: Any) -> str:
        if v is None:
            return "Community Summary"
        s = str(v).strip()
        return s if s else "Community Summary"

    @field_validator("summary", mode="before")
    @classmethod
    def _normalize_summary(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return str(v).strip()

    @field_validator("impact_rating", mode="before")
    @classmethod
    def _normalize_impact_rating(cls, v: Any) -> float:
        if v is None:
            return 5.0
        val = 5.0
        if isinstance(v, (int, float)):
            val = float(v)
        elif isinstance(v, str):
            match = re.search(r"(\d+(?:\.\d+)?)", v)
            if match:
                try:
                    val = float(match.group(1))
                except ValueError:
                    val = 5.0
        else:
            try:
                val = float(v)
            except (ValueError, TypeError):
                val = 5.0
        if math.isnan(val) or math.isinf(val):
            val = 5.0
        return max(1.0, min(10.0, val))

    @field_validator("rating_explanation", mode="before")
    @classmethod
    def _normalize_rating_explanation(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("findings", mode="before")
    @classmethod
    def _normalize_findings(cls, v: Any) -> List[Dict[str, Any]]:
        if v is None:
            return []
        if isinstance(v, dict):
            v = [v]
        elif isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    v = parsed
                elif isinstance(parsed, dict):
                    v = [parsed]
                else:
                    v = [{"summary": v.strip(), "explanation": ""}]
            except Exception:
                # Check for bullet list in string
                stripped = v.strip()
                raw_lines = [
                    ln.strip().lstrip("-* \t").strip()
                    for ln in stripped.split("\n")
                    if ln.strip()
                ]
                if len(raw_lines) > 1:
                    v = [
                        {"summary": ln, "explanation": ""}
                        for ln in raw_lines
                    ]
                else:
                    v = [{"summary": stripped, "explanation": ""}]
        elif not isinstance(v, list):
            return []

        normalized = []
        for item in v:
            if isinstance(item, dict):
                d = {str(k): val for k, val in item.items()}
                if "summary" not in d:
                    d["summary"] = str(
                        d.get("title")
                        or d.get("finding")
                        or d.get("name")
                        or d.get("claim")
                        or "Key Finding"
                    )
                if "explanation" not in d:
                    d["explanation"] = str(
                        d.get("description")
                        or d.get("detail")
                        or d.get("evidence")
                        or ""
                    )
                normalized.append(d)
            elif isinstance(item, str):
                normalized.append({"summary": item.strip(), "explanation": ""})
            else:
                normalized.append({"summary": str(item), "explanation": ""})
        return normalized


class CommunitySummarizer:
    """
    Engine for generating executive community reports in GraphRAG pipelines.

    Features:
        - Multi-tier LLM unwrapping across SDK wrappers and callables
        - Centrality-based deterministic token budgeting and context packing
        - Thread-safe SHA-256 content caching with atomic disk persistence
        - Subgraph extraction fallback when raw graph instance is unavailable
        - Bottom-up hierarchical synthesis across coarsening levels
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_tokens: int = 4000,
        token_counter: Optional[Callable[[str], int]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        centrality_calculator: Optional[CentralityCalculator] = None,
        centrality_metric: str = "degree",
        cache_enabled: bool = True,
        embedder: Optional[Callable[[str], List[float]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.logger = get_logger("community_summarizer")
        self.llm = llm
        self.max_tokens = max(200, int(max_tokens))
        self.token_counter = token_counter
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.centrality_calculator = centrality_calculator
        self.centrality_metric = centrality_metric
        self.cache_enabled = bool(cache_enabled)
        self.embedder = embedder
        self.system_prompt = system_prompt
        self.config = kwargs

        self._memory_cache: Dict[str, CommunityReport] = {}
        self._lock = threading.Lock()

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cache_key: str) -> Optional[Path]:
        """Generate safe, traversal-proof cache file path."""
        if self.cache_dir is None or not cache_key:
            return None
        safe_key = re.sub(r"[^\w\-]", "_", str(cache_key))
        return self.cache_dir / f"{safe_key}.json"

    def get_cached_report(self, cache_key: str) -> Optional[CommunityReport]:
        """Retrieve a cached community report by SHA-256 content hash."""
        if not cache_key or not self.cache_enabled:
            return None

        with self._lock:
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]

            cache_file = self._cache_path(cache_key)
            if cache_file is not None and cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    report = CommunityReport.from_dict(data)
                    self._memory_cache[cache_key] = report
                    return report
                except Exception as e:
                    self.logger.warning(
                        f"Failed to read cache file {cache_file}: {e}"
                    )
        return None

    def cache_report(self, cache_key: str, report: CommunityReport) -> None:
        """Store a community report in cache with atomic disk persistence."""
        if not cache_key or not self.cache_enabled:
            return

        with self._lock:
            self._memory_cache[cache_key] = report

            cache_file = self._cache_path(cache_key)
            if cache_file is not None and self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=str(self.cache_dir),
                        delete=False,
                        suffix=".tmp",
                    ) as tf:
                        json.dump(
                            report.to_dict(), tf, indent=2, default=str
                        )
                        temp_path = tf.name
                    os.replace(temp_path, cache_file)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to persist cache file {cache_file}: {e}"
                    )
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass

    def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry from memory and disk."""
        found = False
        with self._lock:
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                found = True

            cache_file = self._cache_path(cache_key)
            if cache_file is not None and cache_file.exists():
                try:
                    cache_file.unlink()
                    found = True
                except OSError as e:
                    self.logger.warning(
                        f"Failed to remove cache file {cache_file}: {e}"
                    )
        return found

    def clear_cache(self) -> None:
        """Clear all in-memory and disk cache entries."""
        with self._lock:
            self._memory_cache.clear()
            if self.cache_dir is not None and self.cache_dir.exists():
                for json_file in self.cache_dir.glob("*.json"):
                    try:
                        json_file.unlink()
                    except OSError:
                        pass

    def _extract_subgraph(
        self,
        community: HierarchicalCommunity,
        graph: Optional[Any] = None,
    ) -> Any:
        """Extract community subgraph with fallback to entity_ids and edges."""
        node_set: Set[str] = set(str(e) for e in community.entity_ids)

        if graph is not None:
            if isinstance(graph, CommunityHierarchy):
                try:
                    return graph.get_subgraph(community)
                except Exception as e:
                    self.logger.debug(
                        f"CommunityHierarchy.get_subgraph failed: {e}"
                    )

            if hasattr(graph, "subgraph") and callable(graph.subgraph):
                try:
                    matching = [
                        n for n in graph.nodes
                        if str(n) in node_set or n in node_set
                    ]
                    return graph.subgraph(matching).copy()
                except Exception as e:
                    self.logger.debug(f"graph.subgraph failed: {e}")

            if hasattr(graph, "entities") and hasattr(graph, "relationships"):
                try:
                    ents = [
                        e for e in graph.entities
                        if (
                            str(e.get("id", ""))
                            if isinstance(e, dict)
                            else str(e)
                        ) in node_set
                    ]
                    rels = [
                        r for r in graph.relationships
                        if (
                            str(r.get("source", r.get("source_id", "")))
                            if isinstance(r, dict)
                            else str(r[0])
                        ) in node_set
                        and (
                            str(r.get("target", r.get("target_id", "")))
                            if isinstance(r, dict)
                            else str(r[1])
                        ) in node_set
                    ]
                    return {"entities": ents, "relationships": rels}
                except Exception as e:
                    self.logger.debug(f"KnowledgeGraph filtering failed: {e}")

            if isinstance(graph, dict) and (
                "entities" in graph or "relationships" in graph
            ):
                ents = [
                    e for e in graph.get("entities", [])
                    if (
                        str(e.get("id", "")) if isinstance(e, dict) else str(e)
                    ) in node_set
                ]
                rels = [
                    r for r in graph.get("relationships", [])
                    if (
                        str(r.get("source", r.get("source_id", "")))
                        if isinstance(r, dict)
                        else str(r[0])
                    ) in node_set
                    and (
                        str(r.get("target", r.get("target_id", "")))
                        if isinstance(r, dict)
                        else str(r[1])
                    ) in node_set
                ]
                return {"entities": ents, "relationships": rels}

        # Subgraph extraction fallback when graph is None or unhandled
        try:
            import networkx as nx

            nx_graph = nx.DiGraph() if community.directed else nx.Graph()
            nx_graph.add_nodes_from(community.entity_ids)
            for edge in community.edges:
                src = edge.get("source")
                tgt = edge.get("target")
                if src is not None and tgt is not None:
                    attrs = edge.get("attributes") or {}
                    nx_graph.add_edge(str(src), str(tgt), **attrs)
            return nx_graph
        except (ImportError, Exception):
            return {
                "entities": [
                    {"id": str(e), "name": str(e)}
                    for e in community.entity_ids
                ],
                "relationships": [
                    {
                        "source": str(edge.get("source", "")),
                        "target": str(edge.get("target", "")),
                        "type": str(
                            (edge.get("attributes") or {}).get(
                                "type", "CONNECTED_TO"
                            )
                        ),
                        **(edge.get("attributes") or {}),
                    }
                    for edge in community.edges
                ],
            }

    def _compute_centrality(
        self,
        subgraph: Any,
        entity_ids: List[str],
    ) -> Dict[str, float]:
        """Compute centrality scores, handling non-NetworkX subgraphs."""
        scores: Dict[str, float] = {}
        if not entity_ids:
            return scores

        try:
            calculator = (
                self.centrality_calculator
                if self.centrality_calculator is not None
                else CentralityCalculator()
            )
            metric_fn_name = f"calculate_{self.centrality_metric}_centrality"
            calc_func = getattr(
                calculator,
                metric_fn_name,
                calculator.calculate_degree_centrality,
            )
            res = calc_func(subgraph)
            if isinstance(res, dict):
                cent_dict = (
                    res["centrality"]
                    if (
                        "centrality" in res
                        and isinstance(res["centrality"], dict)
                    )
                    else res
                )
                if isinstance(cent_dict, dict):
                    for k, v in cent_dict.items():
                        try:
                            scores[str(k)] = float(v)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            self.logger.debug(f"Centrality calculation error: {e}")

        for eid in entity_ids:
            if str(eid) not in scores:
                scores[str(eid)] = 0.0

        return scores

    def _identify_bridge_edges(
        self,
        community: HierarchicalCommunity,
        child_reports: Optional[List[CommunityReport]] = None,
        subgraph: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Identify bridge edges connecting different sub-communities."""
        raw_edges = list(community.edges)
        if not raw_edges and subgraph is not None:
            if hasattr(subgraph, "edges"):
                try:
                    for u, v, d in subgraph.edges(data=True):
                        d_dict = dict(d) if isinstance(d, dict) else {}
                        raw_edges.append(
                            {
                                "source": str(u),
                                "target": str(v),
                                "type": str(
                                    d_dict.get("type", "CONNECTED_TO")
                                ),
                                "attributes": d_dict,
                            }
                        )
                except Exception as e:
                    self.logger.debug(
                        f"Failed extracting edges from subgraph: {e}"
                    )
            elif isinstance(subgraph, dict) and "relationships" in subgraph:
                for rel in subgraph["relationships"]:
                    if isinstance(rel, dict):
                        raw_edges.append(rel)
            elif hasattr(subgraph, "relationships"):
                for rel in subgraph.relationships:
                    if isinstance(rel, dict):
                        raw_edges.append(rel)

        if not raw_edges or not child_reports:
            return raw_edges

        node_to_child: Dict[str, str] = {}
        for cr in child_reports:
            for ent in cr.member_entities:
                node_to_child[str(ent)] = str(cr.community_id)

        bridge_edges = []
        internal_edges = []
        for e in raw_edges:
            src = str(e.get("source", ""))
            tgt = str(e.get("target", ""))
            src_child = node_to_child.get(src)
            tgt_child = node_to_child.get(tgt)
            if src_child and tgt_child and src_child != tgt_child:
                bridge_edges.append(e)
            else:
                internal_edges.append(e)

        if bridge_edges:
            return bridge_edges + internal_edges
        return internal_edges

    def _pack_context(
        self,
        community: HierarchicalCommunity,
        subgraph: Any,
        child_reports: Optional[List[CommunityReport]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Pack community context within token budget using centrality and impact.

        For level >= 1, allocates at most 50% to child reports sorted by
        (-impact_rating, str(id)) and remaining to bridge edges and
        anchor entities.
        """
        budget = max_tokens if max_tokens is not None else self.max_tokens

        prompt_overhead = 400
        available_budget = max(100, budget - prompt_overhead)

        scores = self._compute_centrality(subgraph, community.entity_ids)
        sorted_entities = sorted(
            community.entity_ids,
            key=lambda e: (-scores.get(str(e), 0.0), str(e)),
        )

        child_reports_budget = 0
        packed_child_sections: List[str] = []
        tokens_used_children = 0

        if community.level >= 1 and child_reports:
            child_reports_budget = int(available_budget * 0.50)
            sorted_child_reports = sorted(
                child_reports,
                key=lambda r: (-float(r.impact_rating), str(r.community_id)),
            )

            for cr in sorted_child_reports:
                cr_text = (
                    f"### Sub-Community {cr.community_id} (Level {cr.level}, "
                    f"Impact: {cr.impact_rating:.1f}/10): {cr.title}\n"
                    f"{cr.summary}\n"
                )
                if cr.findings:
                    bullets = []
                    for f in cr.findings[:3]:
                        if isinstance(f, dict):
                            s = (
                                f.get("summary")
                                or f.get("title")
                                or f.get("finding")
                                or f.get("name")
                                or f.get("claim")
                                or ""
                            )
                            e = (
                                f.get("explanation")
                                or f.get("description")
                                or f.get("detail")
                                or f.get("evidence")
                                or ""
                            )
                            bullets.append(f"{s}: {e}".strip(": "))
                        else:
                            bullets.append(str(f))
                    if bullets:
                        cr_text += (
                            "Key Findings:\n- "
                            + "\n- ".join(bullets)
                            + "\n"
                        )

                t_count = estimate_tokens(cr_text, self.token_counter)
                if tokens_used_children + t_count <= child_reports_budget:
                    packed_child_sections.append(cr_text)
                    tokens_used_children += t_count
                else:
                    break

        remaining_budget = available_budget - tokens_used_children

        bridge_edges = self._identify_bridge_edges(
            community, child_reports, subgraph=subgraph
        )
        bridge_edges.sort(
            key=lambda e: (
                min(str(e.get("source", "")), str(e.get("target", ""))),
                max(str(e.get("source", "")), str(e.get("target", ""))),
                json.dumps(
                    e.get("attributes") or {},
                    sort_keys=True,
                    default=str,
                ),
            )
        )

        edge_budget = int(remaining_budget * 0.45)
        packed_edges: List[str] = []
        tokens_used_edges = 0

        for edge in bridge_edges:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            attrs = edge.get("attributes") or {}
            rel_type = (
                edge.get("type")
                or attrs.get("type")
                or "CONNECTED_TO"
            )
            edge_line = f"- ({src}) -[{rel_type}]-> ({tgt})\n"
            t_count = estimate_tokens(edge_line, self.token_counter)
            if tokens_used_edges + t_count <= edge_budget:
                packed_edges.append(edge_line)
                tokens_used_edges += t_count
            else:
                break

        entity_budget = remaining_budget - tokens_used_edges
        packed_entities: List[str] = []
        tokens_used_entities = 0

        for ent in sorted_entities:
            score = scores.get(str(ent), 0.0)
            ent_line = f"- {ent} (centrality: {score:.3f})\n"
            t_count = estimate_tokens(ent_line, self.token_counter)
            if tokens_used_entities + t_count <= entity_budget:
                packed_entities.append(ent_line)
                tokens_used_entities += t_count
            else:
                break

        sections = [
            f"Community ID: {community.id}",
            f"Level: {community.level}",
            f"Total Member Entities: {len(community.entity_ids)}",
        ]

        if packed_child_sections:
            sections.append(
                "## Child Community Reports\n"
                + "\n".join(packed_child_sections)
            )

        if packed_entities:
            sections.append(
                "## Anchor Entities (ranked by centrality)\n"
                + "".join(packed_entities)
            )

        if packed_edges:
            sections.append(
                "## Key Relationships / Bridge Edges\n"
                + "".join(packed_edges)
            )

        return "\n\n".join(sections)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON object from LLM response text."""
        cleaned = text.strip()
        try:
            val = json.loads(cleaned)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # Try markdown code fences ```json ... ```
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL
        )
        if match:
            block = match.group(1).strip()
            try:
                val = json.loads(block)
                if isinstance(val, dict):
                    return val
            except Exception:
                decoder = json.JSONDecoder()
                for i in range(len(block)):
                    if block[i] == "{":
                        try:
                            obj, _ = decoder.raw_decode(block[i:])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass

        # Scan text for first valid JSON object using raw_decode
        decoder = json.JSONDecoder()
        for i in range(len(cleaned)):
            if cleaned[i] == "{":
                try:
                    obj, _ = decoder.raw_decode(cleaned[i:])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

        raise ValueError(
            f"No valid JSON found in LLM output: {cleaned[:120]}..."
        )

    def _schema_from_freeform_text(
        self, text: str, community: HierarchicalCommunity
    ) -> CommunityReportLLMSchema:
        """Create fallback schema when LLM returns plain freeform text."""
        cleaned = text.strip()
        title = f"Community {community.id} Summary"
        findings = [
            {
                "summary": f"Insights for community {community.id}",
                "explanation": cleaned[:250],
            }
        ]
        return CommunityReportLLMSchema(
            title=title,
            summary=cleaned if cleaned else "No summary available.",
            findings=findings,
            impact_rating=5.0,
            rating_explanation="Generated from freeform text response.",
        )

    def _extractive_fallback(
        self, community: HierarchicalCommunity
    ) -> CommunityReportLLMSchema:
        """Deterministic extractive baseline when no LLM is provided."""
        preview = ", ".join(sorted(str(e) for e in community.entity_ids)[:8])
        title = f"Community {community.id} (Level {community.level})"
        summary = (
            f"Community {community.id} contains "
            f"{len(community.entity_ids)} entities including: {preview}."
        )
        findings = [
            {
                "summary": f"Cluster of {len(community.entity_ids)} entities",
                "explanation": (
                    f"Entities in community {community.id} form an "
                    f"interconnected sub-network at level {community.level}."
                ),
            }
        ]
        return CommunityReportLLMSchema(
            title=title,
            summary=summary,
            findings=findings,
            impact_rating=5.0,
            rating_explanation="Extractive baseline report.",
        )

    def _coerce_to_schema(
        self,
        res: Any,
        community: HierarchicalCommunity,
    ) -> Optional[CommunityReportLLMSchema]:
        """Coerce arbitrary response into CommunityReportLLMSchema."""
        if isinstance(res, CommunityReportLLMSchema):
            return res
        if isinstance(res, dict):
            return CommunityReportLLMSchema.model_validate(res)
        if hasattr(res, "model_dump") and callable(res.model_dump):
            return CommunityReportLLMSchema.model_validate(res.model_dump())
        if hasattr(res, "__dict__"):
            try:
                return CommunityReportLLMSchema.model_validate(vars(res))
            except Exception:
                pass
        if isinstance(res, str):
            try:
                parsed = self._extract_json(res)
                return CommunityReportLLMSchema.model_validate(parsed)
            except Exception:
                return self._schema_from_freeform_text(res, community)
        return None

    def _call_llm(
        self,
        prompt: str,
        community: HierarchicalCommunity,
        **kwargs: Any,
    ) -> CommunityReportLLMSchema:
        """Invoke LLM via multi-tier unwrap strategy."""
        llm = self.llm
        if llm is None:
            return self._extractive_fallback(community)

        # Tier 1: llm.generate_typed
        if hasattr(llm, "generate_typed") and callable(llm.generate_typed):
            try:
                try:
                    res = llm.generate_typed(
                        prompt, schema=CommunityReportLLMSchema, **kwargs
                    )
                except TypeError:
                    res = llm.generate_typed(
                        prompt, schema=CommunityReportLLMSchema
                    )
                schema = self._coerce_to_schema(res, community)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 1 generate_typed failed: {e}")

        # Tier 2: llm.provider.generate_typed (e.g. semantica.llms.OpenAI)
        if (
            hasattr(llm, "provider")
            and hasattr(llm.provider, "generate_typed")
            and callable(llm.provider.generate_typed)
        ):
            try:
                try:
                    res = llm.provider.generate_typed(
                        prompt, schema=CommunityReportLLMSchema, **kwargs
                    )
                except TypeError:
                    res = llm.provider.generate_typed(
                        prompt, schema=CommunityReportLLMSchema
                    )
                schema = self._coerce_to_schema(res, community)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(
                    f"Tier 2 provider.generate_typed failed: {e}"
                )

        # Tier 3: llm.generate_structured
        if hasattr(llm, "generate_structured") and callable(
            llm.generate_structured
        ):
            try:
                try:
                    res = llm.generate_structured(prompt, **kwargs)
                except TypeError:
                    res = llm.generate_structured(prompt)
                schema = self._coerce_to_schema(res, community)
                if schema is not None:
                    return schema
                if isinstance(res, list) and res and isinstance(res[0], dict):
                    return CommunityReportLLMSchema.model_validate(res[0])
            except Exception as e:
                self.logger.warning(f"Tier 3 generate_structured failed: {e}")

        # Tier 4: llm.generate with regex/JSON parsing
        if hasattr(llm, "generate") and callable(llm.generate):
            try:
                try:
                    text_res = llm.generate(prompt, **kwargs)
                except TypeError:
                    text_res = llm.generate(prompt)
                schema = self._coerce_to_schema(text_res, community)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 4 generate failed: {e}")

        # Tier 5: callable(llm)
        if callable(llm):
            try:
                try:
                    res = llm(prompt, **kwargs)
                except TypeError:
                    res = llm(prompt)
                schema = self._coerce_to_schema(res, community)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 5 callable failed: {e}")

        self.logger.warning(
            "All LLM tiers failed; returning extractive fallback."
        )
        return self._extractive_fallback(community)

    def summarize_community(
        self,
        community: Union[HierarchicalCommunity, Dict[str, Any]],
        graph: Optional[Any] = None,
        child_reports: Optional[List[CommunityReport]] = None,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CommunityReport:
        """
        Generate a structured community report for a single community.

        Args:
            community: HierarchicalCommunity or dict representation.
            graph: Optional graph instance or CommunityHierarchy.
            child_reports: Sub-community reports for hierarchical synthesis.
            use_cache: If True, checks and updates cache.
            max_tokens: Override context token budget.
            **kwargs: Extra parameters passed to LLM generation.

        Returns:
            CommunityReport object.
        """
        if isinstance(community, dict):
            comm_data = dict(community)
            if "id" not in comm_data:
                comm_data["id"] = str(
                    comm_data.get("community_id", "c_0")
                )
            if "level" not in comm_data:
                comm_data["level"] = 0
            if "index" not in comm_data:
                comm_data["index"] = 0
            comm = HierarchicalCommunity.from_dict(comm_data)
        elif isinstance(community, HierarchicalCommunity):
            comm = community
        else:
            raise TypeError(
                "community must be a HierarchicalCommunity or dict"
            )

        content_hash = comm.content_hash
        if not content_hash:
            content_hash = compute_community_hash(
                comm.level,
                comm.index,
                comm.entity_ids,
                comm.child_ids,
                edges=comm.edges,
                directed=comm.directed,
            )
            comm.content_hash = content_hash

        if use_cache and self.cache_enabled:
            cached = self.get_cached_report(content_hash)
            if cached is not None:
                return cached

        subgraph = self._extract_subgraph(comm, graph)
        context_text = self._pack_context(
            comm,
            subgraph,
            child_reports=child_reports,
            max_tokens=max_tokens,
        )

        system_instruction = self.system_prompt or (
            "You are an AI intelligence assistant summarizing knowledge graph "
            "communities into structured GraphRAG reports. Produce a JSON "
            "object with 'title', 'summary', 'findings' "
            "(list of {summary, explanation}), "
            "'impact_rating' (float 1.0 to 10.0), and 'rating_explanation'."
        )

        full_prompt = (
            f"{system_instruction}\n\n"
            f"Context Information:\n{context_text}\n\n"
            "Return ONLY the structured JSON report."
        )

        llm_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in (
                "rank",
                "embedding",
                "use_cache",
                "child_reports",
                "max_tokens",
            )
        }
        schema = self._call_llm(full_prompt, comm, **llm_kwargs)

        rank = kwargs.get("rank")
        if rank is None:
            size_weight = 1.0 + math.log10(max(1, len(comm.entity_ids)))
            rank = round(float(schema.impact_rating * size_weight), 3)
        else:
            rank = float(rank)

        embedding = kwargs.get("embedding")
        if embedding is None and self.embedder is not None:
            try:
                embedding = self.embedder(
                    f"{schema.title}\n\n{schema.summary}"
                )
            except Exception as e:
                self.logger.warning(f"Embedder failed: {e}")

        sub_comms = list(comm.child_ids)
        if not sub_comms and child_reports:
            sub_comms = [str(cr.community_id) for cr in child_reports]

        report = CommunityReport(
            community_id=str(comm.id),
            level=int(comm.level),
            title=schema.title,
            summary=schema.summary,
            findings=schema.findings,
            impact_rating=schema.impact_rating,
            rating_explanation=schema.rating_explanation,
            member_entities=list(comm.entity_ids),
            content_hash=content_hash,
            sub_communities=sub_comms,
            parent_id=comm.parent_id,
            rank=rank,
            embedding=embedding,
            metadata={
                "size": comm.size,
                "directed": comm.directed,
                **dict(comm.metrics),
            },
        )

        if use_cache and self.cache_enabled:
            self.cache_report(content_hash, report)

        return report

    def summarize_hierarchy(
        self,
        hierarchy: CommunityHierarchy,
        graph: Optional[Any] = None,
        levels: Optional[List[int]] = None,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, CommunityReport]:
        """
        Synthesize community reports bottom-up across a multi-level hierarchy.

        Args:
            hierarchy: CommunityHierarchy container.
            graph: Optional graph instance.
            levels: Optional subset of hierarchy levels to summarize.
            use_cache: If True, uses SHA-256 caching.
            max_tokens: Override context token budget.
            **kwargs: Extra arguments passed to single community summarization.

        Returns:
            Dictionary mapping community IDs to CommunityReport objects.
        """
        if hierarchy.is_empty:
            return {}

        all_levels = sorted(hierarchy.levels)
        target_levels = (
            set(levels) if levels is not None else set(all_levels)
        )

        reports: Dict[str, CommunityReport] = {}
        target_graph = (
            graph if graph is not None else getattr(hierarchy, "_graph", None)
        )
        if target_graph is None:
            target_graph = hierarchy

        comm_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("child_reports", "levels")
        }

        # Bottom-up synthesis: process levels in order (0 -> max_level)
        for lvl in all_levels:
            communities = hierarchy.get_communities_at_level(lvl)
            for comm in communities:
                child_reps = [
                    reports[cid] for cid in comm.child_ids if cid in reports
                ]

                report = self.summarize_community(
                    community=comm,
                    graph=target_graph,
                    child_reports=child_reps,
                    use_cache=use_cache,
                    max_tokens=max_tokens,
                    **comm_kwargs,
                )
                reports[comm.id] = report

        if levels is not None:
            return {
                cid: rep
                for cid, rep in reports.items()
                if rep.level in target_levels
            }

        return reports
