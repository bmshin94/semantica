"""
Global Graph Retrieval Engine for Hierarchical GraphRAG.

Implements Map-Reduce query retrieval over hierarchical community reports,
dynamic level selection, token budgeting, parallel Map execution, and
dual-tier executive synthesis with explicit citations.
"""

from collections.abc import Callable
import concurrent.futures
from dataclasses import dataclass, field
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..kg.community_hierarchy import CommunityHierarchy
from ..kg.community_summarizer import CommunityReport, estimate_tokens
from ..utils.logging import get_logger
from .context_retriever import RetrievedContext

logger = get_logger("global_retriever")


@dataclass
class MapKeyPoint:
    """Intermediate key finding extracted during the Map phase."""

    point: str
    description: str = ""
    relevance_score: float = 5.0
    community_id: str = ""
    level: int = 0
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.point = str(self.point).strip()
        self.description = str(self.description).strip()
        self.community_id = str(self.community_id).strip()
        self.level = int(self.level)
        try:
            r_val = float(self.relevance_score)
            if math.isnan(r_val) or math.isinf(r_val):
                r_val = 5.0
        except (ValueError, TypeError):
            r_val = 5.0
        self.relevance_score = max(0.0, min(10.0, r_val))

        if self.entities:
            self.entities = sorted(set(str(e).strip() for e in self.entities if e))
        else:
            self.entities = []

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert MapKeyPoint to dictionary."""
        return {
            "point": self.point,
            "description": self.description,
            "relevance_score": self.relevance_score,
            "community_id": self.community_id,
            "level": self.level,
            "entities": self.entities,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MapKeyPoint":
        """Reconstruct MapKeyPoint from dictionary."""
        return cls(
            point=d.get("point", ""),
            description=d.get("description", ""),
            relevance_score=float(d.get("relevance_score", 5.0)),
            community_id=str(d.get("community_id", "")),
            level=int(d.get("level", 0)),
            entities=list(d.get("entities", [])),
            metadata=dict(d.get("metadata", {})),
        )


class MapPointSchema(BaseModel):
    """Pydantic schema for individual mapped key points."""

    model_config = ConfigDict(extra="ignore")

    point: str = Field(description="Key finding or core assertion")
    description: str = Field(
        default="", description="Supporting explanation and context"
    )
    relevance_score: float = Field(
        default=5.0, description="Relevance to query (0.0 to 10.0)"
    )
    entities: List[str] = Field(
        default_factory=list, description="Key entities mentioned"
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Clamp relevance score to 0.0 - 10.0."""
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return 5.0
            return max(0.0, min(10.0, val))
        except (ValueError, TypeError):
            return 5.0

    @field_validator("entities", mode="before")
    @classmethod
    def normalize_entities(cls, v: Any) -> List[str]:
        """Normalize entities list."""
        if isinstance(v, list):
            return [str(item).strip() for item in v if item]
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return []


class MapResponseSchema(BaseModel):
    """Pydantic v2 schema for structured Map phase LLM output."""

    model_config = ConfigDict(extra="ignore")

    points: List[MapPointSchema] = Field(
        default_factory=list, description="List of key points extracted"
    )
    relevance_explanation: str = Field(
        default="", description="Rationale for relevance evaluations"
    )

    @field_validator("points", mode="before")
    @classmethod
    def normalize_points(cls, v: Any) -> List[Any]:
        """Normalize flexible point representations into valid schema items."""
        if not isinstance(v, list):
            if isinstance(v, (dict, str)):
                v = [v]
            else:
                return []

        normalized = []
        for item in v:
            if isinstance(item, MapPointSchema):
                normalized.append(item)
            elif isinstance(item, dict):
                p_text = (
                    item.get("point")
                    or item.get("finding")
                    or item.get("summary")
                    or item.get("title")
                    or item.get("claim")
                    or ""
                )
                desc = (
                    item.get("description")
                    or item.get("explanation")
                    or item.get("detail")
                    or item.get("evidence")
                    or ""
                )
                score = (
                    item.get("relevance_score")
                    if item.get("relevance_score") is not None
                    else item.get("score", 5.0)
                )
                ents = item.get("entities") or item.get("member_entities") or []
                normalized.append(
                    {
                        "point": str(p_text),
                        "description": str(desc),
                        "relevance_score": score,
                        "entities": ents,
                    }
                )
            elif isinstance(item, str):
                normalized.append(
                    {
                        "point": item.strip(),
                        "description": "",
                        "relevance_score": 5.0,
                        "entities": [],
                    }
                )
        return normalized


@dataclass
class GlobalSearchResult:
    """Result of global hierarchical GraphRAG search."""

    query: str
    response: str
    level: int = 0
    key_points: List[MapKeyPoint] = field(default_factory=list)
    community_reports_used: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_retrieved_contexts(self) -> List[RetrievedContext]:
        """
        Convert global search result into standard RetrievedContext objects.

        The executive response is placed first with score 1.0, followed by
        individual intermediate key points scaled to 0.0 - 1.0. Returns an
        empty list if no community reports or key points are present.
        """
        contexts: List[RetrievedContext] = []

        if not self.community_reports_used and not self.key_points:
            return contexts

        # Primary executive answer
        contexts.append(
            RetrievedContext(
                content=self.response,
                score=1.0,
                source="global_search",
                metadata={
                    "query": self.query,
                    "level": self.level,
                    "community_reports_used": list(self.community_reports_used),
                    "citations": list(self.citations),
                    "metrics": dict(self.metrics),
                },
                related_entities=[],
                related_relationships=[],
            )
        )

        # Supporting key points
        for kp in self.key_points:
            scaled_score = max(0.0, min(1.0, kp.relevance_score / 10.0))
            content_parts = [kp.point]
            if kp.description:
                content_parts.append(kp.description)
            kp_content = ": ".join(content_parts)

            contexts.append(
                RetrievedContext(
                    content=kp_content,
                    score=scaled_score,
                    source=f"community_{kp.community_id}",
                    metadata={
                        "community_id": kp.community_id,
                        "level": kp.level,
                        "relevance_score": kp.relevance_score,
                        **dict(kp.metadata),
                    },
                    related_entities=[
                        {"id": ent, "name": ent} for ent in kp.entities
                    ],
                    related_relationships=[],
                )
            )

        return contexts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GlobalSearchResult to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "level": self.level,
            "key_points": [kp.to_dict() for kp in self.key_points],
            "community_reports_used": list(self.community_reports_used),
            "citations": list(self.citations),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalSearchResult":
        """Deserialize GlobalSearchResult from dictionary."""
        raw_kps = d.get("key_points", [])
        kps = [
            MapKeyPoint.from_dict(kp)
            if isinstance(kp, dict)
            else kp
            for kp in raw_kps
        ]
        return cls(
            query=d.get("query", ""),
            response=d.get("response", ""),
            level=int(d.get("level", 0)),
            key_points=kps,
            community_reports_used=list(d.get("community_reports_used", [])),
            citations=list(d.get("citations", [])),
            metrics=dict(d.get("metrics", {})),
        )


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numerical vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _extract_words(text: str) -> Set[str]:
    """Extract normalized lowercase alphanumeric word tokens from text."""
    if not text:
        return set()
    return set(re.findall(r"\b\w+\b", text.lower()))


class GlobalGraphRetriever:
    """
    Map-Reduce global query retrieval engine for Hierarchical GraphRAG.

    Executes dynamic level promotion, token-budgeted pruning, concurrent Map
    evaluations across community reports, and executive Reduce synthesis with
    attribution.
    """

    def __init__(
        self,
        reports: Optional[
            Union[
                Sequence[Union[CommunityReport, Dict[str, Any]]],
                Dict[str, Any],
            ]
        ] = None,
        hierarchy: Optional[CommunityHierarchy] = None,
        llm: Optional[Any] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_context_tokens: int = 4000,
        response_token_budget: int = 600,
        fixed_overhead: int = 300,
        min_relevance_score: float = 0.0,
        max_workers: int = 4,
        timeout: float = 60.0,
        auto_promote_level: bool = True,
        **kwargs: Any,
    ) -> None:
        self.logger = get_logger("global_retriever")
        self.hierarchy = hierarchy
        self.llm = llm
        self.embedder = embedder
        self.token_counter = token_counter
        self.max_context_tokens = max(1, int(max_context_tokens))
        self.response_token_budget = max(10, int(response_token_budget))
        self.fixed_overhead = max(0, int(fixed_overhead))
        self.min_relevance_score = max(0.0, min(10.0, float(min_relevance_score)))
        self.max_workers = max(1, int(max_workers))
        self.timeout = max(1.0, float(timeout))
        self.auto_promote_level = bool(auto_promote_level)
        self.config = kwargs

        self._reports: List[CommunityReport] = []
        if reports is not None:
            self.set_reports(reports)

    def set_reports(
        self,
        reports: Union[
            Sequence[Union[CommunityReport, Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> None:
        """Update active community reports collection."""
        loaded: List[CommunityReport] = []
        if isinstance(reports, dict):
            # Check if dict is {community_id: report} or single report dict
            if "community_id" in reports and "summary" in reports:
                loaded.append(CommunityReport.from_dict(reports))
            else:
                for rep in reports.values():
                    if isinstance(rep, CommunityReport):
                        loaded.append(rep)
                    elif isinstance(rep, dict):
                        loaded.append(CommunityReport.from_dict(rep))
        elif isinstance(reports, (list, tuple)):
            for rep in reports:
                if isinstance(rep, CommunityReport):
                    loaded.append(rep)
                elif isinstance(rep, dict):
                    loaded.append(CommunityReport.from_dict(rep))

        self._reports = loaded

    def _estimate_report_tokens(self, report: CommunityReport) -> int:
        """Estimate token cost of a single community report."""
        findings_str = " ".join(
            f.get("summary", "") + " " + f.get("explanation", "")
            for f in report.findings
            if isinstance(f, dict)
        )
        full_text = f"{report.title}\n{report.summary}\n{findings_str}"
        return estimate_tokens(full_text, self.token_counter)

    def _group_reports_by_level(self) -> Dict[int, List[CommunityReport]]:
        """Group active reports by coarsening hierarchy level."""
        grouped: Dict[int, List[CommunityReport]] = {}
        for r in self._reports:
            grouped.setdefault(r.level, []).append(r)
        return grouped

    def select_level_and_budget(
        self,
        query: str,
        target_level: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        auto_promote_level: Optional[bool] = None,
    ) -> tuple[int, List[CommunityReport]]:
        """
        Dynamically select hierarchy level and apply token-budgeted pruning.

        Promotes to coarser level L+1 if total tokens at level L exceed budget
        and auto_promote_level is True. If promotion is exhausted or disabled,
        prunes low-relevance reports at the selected level.
        """
        grouped = self._group_reports_by_level()
        if not grouped:
            return (0, [])

        available_levels = sorted(grouped.keys())

        if target_level is not None and target_level in grouped:
            current_level = target_level
        else:
            current_level = available_levels[0]

        effective_auto_promote = (
            self.auto_promote_level
            if auto_promote_level is None
            else bool(auto_promote_level)
        )

        # Dynamic level promotion
        if effective_auto_promote:
            while current_level < available_levels[-1]:
                candidates = grouped.get(current_level, [])
                total_tokens = sum(
                    self._estimate_report_tokens(r) for r in candidates
                )
                if total_tokens <= self.max_context_tokens:
                    break
                # Promote to next available level
                higher_levels = [lvl for lvl in available_levels if lvl > current_level]
                if not higher_levels:
                    break
                current_level = higher_levels[0]

        candidates = grouped.get(current_level, [])
        total_tokens = sum(self._estimate_report_tokens(r) for r in candidates)

        if total_tokens <= self.max_context_tokens:
            return (current_level, candidates)

        # Budget exceeded: prune candidate reports at current_level
        if query_embedding is None and self.embedder is not None:
            try:
                query_embedding = self.embedder(query)
            except Exception as e:
                self.logger.warning(f"Query embedding generation failed: {e}")

        scored_candidates: List[tuple[float, CommunityReport]] = []
        query_words = _extract_words(query)

        for rep in candidates:
            score = 0.0
            if (
                query_embedding is not None
                and rep.embedding is not None
                and len(rep.embedding) == len(query_embedding)
            ):
                score = _cosine_similarity(query_embedding, rep.embedding)
            else:
                findings_gen = (
                    str(f.get("summary", ""))
                    for f in rep.findings
                    if isinstance(f, dict)
                )
                rep_text = f"{rep.title} {rep.summary} " + " ".join(findings_gen)
                rep_words = _extract_words(rep_text)
                if query_words and rep_words:
                    intersection = len(query_words & rep_words)
                    union = len(query_words | rep_words)
                    jaccard = intersection / float(union) if union > 0 else 0.0
                else:
                    jaccard = 0.0

                norm_rank = max(0.0, min(1.0, rep.rank))
                norm_impact = max(0.0, min(1.0, (rep.impact_rating - 1.0) / 9.0))
                score = (jaccard * 0.5) + (norm_rank * 0.3) + (norm_impact * 0.2)

            scored_candidates.append((score, rep))

        # Sort descending by score, tie-break by impact_rating and community_id
        scored_candidates.sort(
            key=lambda item: (-item[0], -item[1].impact_rating, item[1].community_id)
        )

        retained: List[CommunityReport] = []
        accumulated_tokens = 0
        for _, rep in scored_candidates:
            cost = self._estimate_report_tokens(rep)
            if accumulated_tokens + cost <= self.max_context_tokens or not retained:
                retained.append(rep)
                accumulated_tokens += cost
            else:
                continue

        return (current_level, retained)

    def _extract_json(self, text: str) -> Union[Dict[str, Any], List[Any]]:
        """Extract and parse JSON from LLM response."""
        cleaned = text.strip()
        try:
            val = json.loads(cleaned)
            if isinstance(val, (dict, list)):
                return val
        except Exception:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
        if match:
            block = match.group(1).strip()
            try:
                val = json.loads(block)
                if isinstance(val, (dict, list)):
                    return val
            except Exception:
                pass

        decoder = json.JSONDecoder()
        for i in range(len(cleaned)):
            if cleaned[i] in ("{", "["):
                try:
                    obj, _ = decoder.raw_decode(cleaned[i:])
                    if isinstance(obj, (dict, list)):
                        return obj
                except Exception:
                    pass

        raise ValueError(f"No valid JSON found in LLM output: {cleaned[:100]}...")

    def _coerce_map_response(
        self, res: Any, report: CommunityReport
    ) -> Optional[MapResponseSchema]:
        """Coerce raw LLM output to MapResponseSchema."""
        if isinstance(res, MapResponseSchema):
            return res
        if isinstance(res, list):
            return MapResponseSchema(points=res)
        if isinstance(res, dict):
            if "points" not in res and any(
                k in res
                for k in ("point", "finding", "summary", "title", "claim")
            ):
                return MapResponseSchema(points=[res])
            return MapResponseSchema.model_validate(res)
        if hasattr(res, "model_dump") and callable(res.model_dump):
            return MapResponseSchema.model_validate(res.model_dump())
        if hasattr(res, "__dict__"):
            try:
                return MapResponseSchema.model_validate(vars(res))
            except Exception:
                pass
        if isinstance(res, str):
            try:
                parsed = self._extract_json(res)
                if isinstance(parsed, list):
                    return MapResponseSchema(points=parsed)
                elif isinstance(parsed, dict):
                    if "points" not in parsed and any(
                        k in parsed
                        for k in ("point", "finding", "summary", "title", "claim")
                    ):
                        return MapResponseSchema(points=[parsed])
                    return MapResponseSchema.model_validate(parsed)
            except Exception:
                # Freeform text fallback: treat as single point
                cleaned = res.strip()
                if cleaned:
                    return MapResponseSchema(
                        points=[
                            MapPointSchema(
                                point=f"Finding from Community {report.community_id}",
                                description=cleaned[:300],
                                relevance_score=5.0,
                                entities=list(report.member_entities[:5]),
                            )
                        ],
                        relevance_explanation="Extracted from freeform LLM response.",
                    )
        return None

    def _extractive_map_fallback(
        self, report: CommunityReport, query: str
    ) -> MapResponseSchema:
        """Deterministic keyword-based extractive fallback for Map phase."""
        query_words = _extract_words(query)
        points: List[MapPointSchema] = []

        rep_text = f"{report.title} {report.summary}"
        rep_words = _extract_words(rep_text)
        overlap = len(query_words & rep_words) if query_words else 0
        base_score = 5.0 + min(5.0, overlap * 1.5) if query_words else 5.0

        if report.summary:
            points.append(
                MapPointSchema(
                    point=report.title or f"Community {report.community_id} Overview",
                    description=report.summary,
                    relevance_score=base_score,
                    entities=list(report.member_entities[:5]),
                )
            )

        for finding in report.findings:
            if isinstance(finding, dict):
                f_summary = (
                    finding.get("summary")
                    or finding.get("title")
                    or finding.get("finding")
                    or ""
                )
                f_expl = (
                    finding.get("explanation")
                    or finding.get("description")
                    or ""
                )
                f_words = _extract_words(f"{f_summary} {f_expl}")
                f_overlap = len(query_words & f_words) if query_words else 0
                f_score = (
                    max(1.0, min(10.0, 4.0 + f_overlap * 2.0))
                    if query_words
                    else 5.0
                )

                points.append(
                    MapPointSchema(
                        point=f_summary or "Community Finding",
                        description=f_expl,
                        relevance_score=f_score,
                        entities=list(report.member_entities[:3]),
                    )
                )

        if not points:
            members_preview = ", ".join(report.member_entities[:5])
            points.append(
                MapPointSchema(
                    point=f"Community {report.community_id} Data",
                    description=f"Contains entities: {members_preview}",
                    relevance_score=5.0,
                    entities=list(report.member_entities[:5]),
                )
            )

        return MapResponseSchema(
            points=points,
            relevance_explanation="Generated via extractive keyword fallback.",
        )

    def _call_map_llm(
        self, prompt: str, report: CommunityReport, **kwargs: Any
    ) -> MapResponseSchema:
        """Execute 6-tier LLM unwrap for Map phase evaluation."""
        llm = kwargs.get("llm") or self.llm
        if llm is None:
            return self._extractive_map_fallback(report, prompt)

        # Tier 1: llm.generate_typed
        tier1_attempted = False
        if hasattr(llm, "generate_typed") and callable(llm.generate_typed):
            tier1_attempted = True
            try:
                try:
                    res = llm.generate_typed(
                        prompt, schema=MapResponseSchema, **kwargs
                    )
                except TypeError:
                    res = llm.generate_typed(prompt, schema=MapResponseSchema)
                schema = self._coerce_map_response(res, report)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 1 generate_typed failed: {e}")

        # Tier 2: llm.provider.generate_typed
        if (
            not tier1_attempted
            and hasattr(llm, "provider")
            and hasattr(llm.provider, "generate_typed")
            and callable(llm.provider.generate_typed)
        ):
            try:
                try:
                    res = llm.provider.generate_typed(
                        prompt, schema=MapResponseSchema, **kwargs
                    )
                except TypeError:
                    res = llm.provider.generate_typed(
                        prompt, schema=MapResponseSchema
                    )
                schema = self._coerce_map_response(res, report)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 2 provider.generate_typed failed: {e}")

        # Tier 3: llm.generate_structured
        if hasattr(llm, "generate_structured") and callable(
            llm.generate_structured
        ):
            try:
                try:
                    res = llm.generate_structured(prompt, **kwargs)
                except TypeError:
                    res = llm.generate_structured(prompt)
                schema = self._coerce_map_response(res, report)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 3 generate_structured failed: {e}")

        # Tier 4: llm.generate + JSON regex
        if hasattr(llm, "generate") and callable(llm.generate):
            try:
                try:
                    text_res = llm.generate(prompt, **kwargs)
                except TypeError:
                    text_res = llm.generate(prompt)
                schema = self._coerce_map_response(text_res, report)
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
                schema = self._coerce_map_response(res, report)
                if schema is not None:
                    return schema
            except Exception as e:
                self.logger.warning(f"Tier 5 callable failed: {e}")

        # Tier 6: Extractive keyword fallback
        self.logger.warning("All LLM tiers failed; using extractive Map fallback.")
        return self._extractive_map_fallback(report, prompt)

    def _map_report(
        self, report: CommunityReport, query: str
    ) -> List[MapKeyPoint]:
        """Execute Map task for a single community report."""
        findings_bullets = []
        for f in report.findings:
            if isinstance(f, dict):
                s = f.get("summary", "")
                e = f.get("explanation", "")
                findings_bullets.append(f"- {s}: {e}".strip(": "))
            else:
                findings_bullets.append(f"- {f}")
        findings_text = "\n".join(findings_bullets) if findings_bullets else "None"

        entities_preview = ", ".join(report.member_entities[:10])

        prompt = (
            "You are evaluating a knowledge graph community report for the "
            f"following user query.\nQuery: {query}\n\n"
            f"Community ID: {report.community_id} (Level {report.level})\n"
            f"Community Title: {report.title}\n"
            f"Summary: {report.summary}\n"
            f"Entities: {entities_preview}\n"
            f"Key Findings:\n{findings_text}\n\n"
            "Instructions:\n"
            "1. Extract key points relevant to the query from this report.\n"
            "2. For each point, assign a relevance_score between 0.0 and 10.0.\n"
            "3. Return response in JSON with 'points' (list of {'point', "
            "'description', 'relevance_score', 'entities'}) and "
            "'relevance_explanation'."
        )

        response_schema = self._call_map_llm(prompt, report)

        key_points: List[MapKeyPoint] = []
        for p in response_schema.points:
            ents = p.entities if p.entities else list(report.member_entities[:5])
            key_points.append(
                MapKeyPoint(
                    point=p.point,
                    description=p.description,
                    relevance_score=p.relevance_score,
                    community_id=report.community_id,
                    level=report.level,
                    entities=ents,
                    metadata={"source_title": report.title},
                )
            )

        return key_points

    def _execute_parallel_map(
        self, reports: List[CommunityReport], query: str
    ) -> List[MapKeyPoint]:
        """Execute Map phase concurrently across reports with worker error isolation."""
        if not reports:
            return []

        all_points: List[MapKeyPoint] = []
        num_workers = min(self.max_workers, len(reports))
        completed_communities: Set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_report = {
                executor.submit(self._map_report, rep, query): rep
                for rep in reports
            }

            try:
                for future in concurrent.futures.as_completed(
                    future_to_report, timeout=self.timeout
                ):
                    rep = future_to_report[future]
                    completed_communities.add(rep.community_id)
                    try:
                        points = future.result()
                        all_points.extend(points)
                    except Exception as exc:
                        self.logger.warning(
                            f"Map worker failed for {rep.community_id}: {exc}; "
                            "falling back to extractive."
                        )
                        fallback_schema = self._extractive_map_fallback(rep, query)
                        for p in fallback_schema.points:
                            all_points.append(
                                MapKeyPoint(
                                    point=p.point,
                                    description=p.description,
                                    relevance_score=p.relevance_score,
                                    community_id=rep.community_id,
                                    level=rep.level,
                                    entities=(
                                        p.entities or list(rep.member_entities[:5])
                                    ),
                                    metadata={"source_title": rep.title},
                                )
                            )
            except concurrent.futures.TimeoutError:
                self.logger.warning(
                    f"Map phase reached timeout ({self.timeout}s); falling back to "
                    "extractive points for unfinished reports."
                )
                for future, rep in future_to_report.items():
                    if rep.community_id not in completed_communities:
                        fallback_schema = self._extractive_map_fallback(rep, query)
                        for p in fallback_schema.points:
                            all_points.append(
                                MapKeyPoint(
                                    point=p.point,
                                    description=p.description,
                                    relevance_score=p.relevance_score,
                                    community_id=rep.community_id,
                                    level=rep.level,
                                    entities=(
                                        p.entities or list(rep.member_entities[:5])
                                    ),
                                    metadata={"source_title": rep.title},
                                )
                            )

        return all_points

    def _pack_reduce_context(
        self, key_points: List[MapKeyPoint], budget: int
    ) -> tuple[str, List[MapKeyPoint]]:
        """Pack sorted key points into Reduce prompt within token budget."""
        packed_lines: List[str] = []
        retained_points: List[MapKeyPoint] = []
        tokens_used = 0

        for kp in key_points:
            ents_str = f" ({', '.join(kp.entities[:3])})" if kp.entities else ""
            line = (
                f"- [Community {kp.community_id}]{ents_str}: {kp.point}. "
                f"{kp.description} (relevance: {kp.relevance_score:.1f}/10)\n"
            )
            cost = estimate_tokens(line, self.token_counter)
            if tokens_used + cost <= budget or not packed_lines:
                packed_lines.append(line)
                retained_points.append(kp)
                tokens_used += cost
            else:
                break

        return ("".join(packed_lines), retained_points)

    def _synthesize_reduce(
        self, query: str, context_text: str, key_points: List[MapKeyPoint]
    ) -> str:
        """Synthesize final executive answer from packed key points."""
        llm = self.llm
        if llm is None:
            # Deterministic synthesis fallback
            grouped_by_comm: Dict[str, List[str]] = {}
            for kp in key_points:
                grouped_by_comm.setdefault(kp.community_id, []).append(
                    f"{kp.point}: {kp.description}".strip(": ")
                )

            lines = [f"Executive Synthesis for query: '{query}'\n"]
            for comm_id, pts in grouped_by_comm.items():
                lines.append(f"From [Community {comm_id}]:")
                for pt in pts:
                    lines.append(f"  • {pt}")
            return "\n".join(lines)

        prompt = (
            "You are an executive knowledge analyst synthesizing global findings.\n\n"
            f"Query: {query}\n\n"
            f"Key Points from Community Reports:\n{context_text}\n\n"
            "Instructions:\n"
            "1. Synthesize a comprehensive, executive-level answer that directly "
            "answers the query.\n"
            "2. You MUST cite the source communities using [Community <id>] whenever "
            "making assertions.\n"
            "3. Mention key entities in parentheses where appropriate.\n"
            "4. Provide a coherent, well-structured response."
        )

        try:
            if hasattr(llm, "generate") and callable(llm.generate):
                res = llm.generate(prompt)
            elif callable(llm):
                res = llm(prompt)
            else:
                res = str(llm)
            if isinstance(res, dict):
                return str(
                    res.get("text")
                    or res.get("response")
                    or res.get("content")
                    or json.dumps(res)
                )
            return str(res).strip()
        except Exception as e:
            self.logger.warning(f"Reduce LLM synthesis failed: {e}")
            # Deterministic fallback
            return (
                f"Global synthesis for '{query}' based on community findings:\n"
                + context_text
            )

    def search(
        self,
        query: str,
        level: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        min_relevance_score: Optional[float] = None,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """
        Execute Map-Reduce global search across community reports.

        Args:
            query: Natural language query.
            level: Target hierarchy level (coarsening level).
            query_embedding: Optional precomputed embedding for the query.
            min_relevance_score: Minimum relevance threshold for key points.
            **kwargs: Additional runtime options.

        Returns:
            GlobalSearchResult containing response, key points, citations, and metrics.
        """
        start_time = time.time()
        effective_min_relevance = (
            self.min_relevance_score
            if min_relevance_score is None
            else max(0.0, min(10.0, float(min_relevance_score)))
        )

        # Level selection and report budgeting
        selected_level, candidate_reports = self.select_level_and_budget(
            query,
            target_level=level,
            query_embedding=query_embedding,
            auto_promote_level=kwargs.get("auto_promote_level"),
        )

        if not candidate_reports:
            return GlobalSearchResult(
                query=query,
                response="No community reports available for global search.",
                level=selected_level,
                key_points=[],
                community_reports_used=[],
                citations=[],
                metrics={
                    "time_taken": time.time() - start_time,
                    "level_used": selected_level,
                    "reports_evaluated": 0,
                    "reports_mapped": 0,
                    "key_points_generated": 0,
                    "key_points_retained": 0,
                },
            )

        # Parallel Map Phase
        raw_key_points = self._execute_parallel_map(candidate_reports, query)

        # Filter key points by relevance cutoff
        filtered_points = [
            p for p in raw_key_points if p.relevance_score >= effective_min_relevance
        ]

        # Sort descending by relevance score, tie-break by community_id and point text
        filtered_points.sort(
            key=lambda p: (-p.relevance_score, str(p.community_id), str(p.point))
        )

        # Reduce phase token budget
        reduce_budget = max(
            100,
            self.max_context_tokens
            - self.fixed_overhead
            - self.response_token_budget,
        )

        packed_context, retained_points = self._pack_reduce_context(
            filtered_points, reduce_budget
        )

        # Executive Reduce synthesis
        response_text = self._synthesize_reduce(query, packed_context, retained_points)

        # Extract citations
        raw_citations = re.findall(
            r"\[Community\s+([A-Za-z0-9_\-]+)\]", response_text, re.IGNORECASE
        )
        citations = sorted(set(str(c) for c in raw_citations))
        if not citations and retained_points:
            citations = sorted(set(kp.community_id for kp in retained_points))

        reports_used = sorted(set(r.community_id for r in candidate_reports))

        duration = time.time() - start_time
        metrics = {
            "time_taken": duration,
            "level_used": selected_level,
            "reports_evaluated": len(candidate_reports),
            "reports_mapped": len(candidate_reports),
            "key_points_generated": len(raw_key_points),
            "key_points_retained": len(retained_points),
            "citations_count": len(citations),
        }

        return GlobalSearchResult(
            query=query,
            response=response_text,
            level=selected_level,
            key_points=retained_points,
            community_reports_used=reports_used,
            citations=citations,
            metrics=metrics,
        )
