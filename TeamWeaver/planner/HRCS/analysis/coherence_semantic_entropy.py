"""
Coherence & Semantic Entropy Evaluator

This module provides metrics and a lightweight pipeline to analyze
- semantic entropy (meaning-level uncertainty)
- step-to-step coherence (transitional entailment / conditional entropy)
- fact accuracy against a provided world_state/world_graph snapshot

Goals
- Be standalone and minimally invasive: can be imported and called from
  the planner loop (e.g., inside or right after `replan`) without changing
  the planning logic.
- Accept pluggable semantic relation scorers (NLI or similarity models) but
  provide conservative heuristics when such models are not available.
- Produce a structured, human-readable report for transparency.

Typical usage in the planner context
- At each `replan` step, collect:
  - candidate_responses: multiple LLM responses if available (e.g., raw vs compare)
  - selected_response: the response actually used to derive actions
  - latest world_state snapshot, current_phase information, and parsed actions
- Call `CoherenceAnalyzer.generate_report(...)`
- Log or persist the returned dict/JSON for debugging or monitoring

Note on dependencies
- This file deliberately avoids importing heavy planner classes to prevent
  circular dependencies. It only relies on standard library and numpy.
- Planner code can pass `callables` for NLI/scorers and `world accessors` if needed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
 
 
 # =============================
 # Types and Protocols
 # =============================
 
class SemanticRelation(Enum):
    EQUIVALENT = "equivalent"
    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    UNRELATED = "unrelated"
 
 
 # A callable to compare two texts and return a semantic relation with a score in [0,1]
SemanticScorer = Callable[[str, str], Tuple[SemanticRelation, float]]
 
 
 # =============================
 # Utilities
 # =============================
 
def _log_safe(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return math.log(x)
 
 
def compute_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy on a discrete distribution defined by probabilities."""
    total = float(sum(probabilities))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p in probabilities:
        if p <= 0:
            continue
        q = p / total
        entropy -= q * _log_safe(q)
    return float(entropy)
 
 
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())
 
 
 # Action line format in planner I/O, e.g.:
 #   Agent_0_Action: Navigate[toy_food_1]
ACTION_LINE_RE = re.compile(r"^\s*Agent_(\d+)_Action:\s*([A-Za-z]+)\[(.*?)\]\s*$")
 
 
def parse_agent_actions_from_response(response: str) -> Dict[int, Tuple[str, Optional[str]]]:
    """
    Parse per-agent high-level actions from an LLM response string.
    Returns a map: agent_uid -> (action_name, action_arg_str | None)
    """
    parsed: Dict[int, Tuple[str, Optional[str]]] = {}
    for line in response.splitlines():
        m = ACTION_LINE_RE.match(line)
        if not m:
            continue
        try:
            agent_uid = int(m.group(1))
            action_name = m.group(2)
            raw_args = m.group(3).strip()
            arg_value = raw_args if raw_args != "" else None
            parsed[agent_uid] = (action_name, arg_value)
        except Exception:
            continue
    return parsed
 
 
 # =============================
 # Default lightweight semantic scorer (fallback)
 # =============================
 
def _default_semantic_scorer(a: str, b: str) -> Tuple[SemanticRelation, float]:
    """
    A conservative fallback scorer using:
    - exact action equivalence if both contain structured agent action lines
    - else, token overlap (Jaccard) on normalized text
    This is a placeholder for real NLI or embedding-based entailment.
    """
    a_norm, b_norm = normalize_text(a), normalize_text(b)

    # If both look like action lists, compare as sets
    a_actions = parse_agent_actions_from_response(a)
    b_actions = parse_agent_actions_from_response(b)
    if a_actions and b_actions:
        a_set = set(a_actions.items())
        b_set = set(b_actions.items())
        if a_set == b_set:
            return SemanticRelation.EQUIVALENT, 1.0
        # Weak entailment: same verbs for overlapping agents
        overlap = 0
        total = max(len(a_actions), len(b_actions))
        for uid, (act, _arg) in a_actions.items():
            if uid in b_actions and b_actions[uid][0] == act:
                overlap += 1
        score = overlap / total if total > 0 else 0.0
        if score >= 0.8:
            return SemanticRelation.ENTAILS, score
        if score <= 0.2:
            return SemanticRelation.UNRELATED, 1.0 - score
        return SemanticRelation.UNRELATED, score

    # Otherwise use token Jaccard
    a_tokens = set(re.findall(r"[a-z0-9_]+", a_norm))
    b_tokens = set(re.findall(r"[a-z0-9_]+", b_norm))
    if not a_tokens and not b_tokens:
        return SemanticRelation.EQUIVALENT, 1.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jaccard = inter / union if union > 0 else 0.0
    if jaccard >= 0.9:
        return SemanticRelation.EQUIVALENT, jaccard
    if jaccard >= 0.6:
        return SemanticRelation.ENTAILS, jaccard
    if jaccard <= 0.1:
        return SemanticRelation.UNRELATED, 1.0 - jaccard
    return SemanticRelation.UNRELATED, jaccard
 
 
 # =============================
 # Clustering by semantic equivalence
 # =============================
 
@dataclass
class Cluster:
    index: int
    members: List[int] = field(default_factory=list)
    representative: Optional[int] = None
 
 
def cluster_semantics(
    texts: Sequence[str],
    scorer: Optional[SemanticScorer] = None,
    entail_threshold: float = 0.75,
    equivalence_threshold: float = 0.9,
) -> Tuple[List[Cluster], List[int]]:
    """
    Agglomerative-style clustering using mutual entailment/equivalence.
    - scorer returns (relation, score). If score >= equivalence_threshold and relation in {EQUIVALENT}, merge into same cluster
    - else if both ways entail with score >= entail_threshold, treat as same meaning cluster
    Returns (clusters, assignment) where assignment[i] is the cluster index for texts[i].
    """
    if scorer is None:
        scorer = _default_semantic_scorer

    n = len(texts)
    if n == 0:
        return [], []
    if n == 1:
        return [Cluster(index=0, members=[0], representative=0)], [0]

    clusters: List[Cluster] = []
    assignment: List[int] = [-1] * n

    def _is_equivalent(i: int, j: int) -> bool:
        rel_ij, s_ij = scorer(texts[i], texts[j])
        rel_ji, s_ji = scorer(texts[j], texts[i])
        if rel_ij == SemanticRelation.EQUIVALENT and s_ij >= equivalence_threshold:
            return True
        if rel_ji == SemanticRelation.EQUIVALENT and s_ji >= equivalence_threshold:
            return True
        if rel_ij in {SemanticRelation.EQUIVALENT, SemanticRelation.ENTAILS} and \
           rel_ji in {SemanticRelation.EQUIVALENT, SemanticRelation.ENTAILS} and \
           min(s_ij, s_ji) >= entail_threshold:
            return True
        return False

    for i in range(n):
        placed = False
        for c in clusters:
            rep = c.representative if c.representative is not None else c.members[0]
            if _is_equivalent(i, rep):
                c.members.append(i)
                assignment[i] = c.index
                placed = True
                break
        if not placed:
            idx = len(clusters)
            clusters.append(Cluster(index=idx, members=[i], representative=i))
            assignment[i] = idx

    return clusters, assignment
 
 
def estimate_cluster_probabilities(
    assignment: Sequence[int],
    sample_weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Estimate p_c via frequency or weighted frequency."""
    if not assignment:
        return []
    k = max(assignment) + 1
    weights = np.ones(len(assignment), dtype=float) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    probs = np.zeros(k, dtype=float)
    for i, cid in enumerate(assignment):
        probs[cid] += float(weights[i])
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs.tolist()
 
 
 # =============================
 # Metric calculators
 # =============================
 
@dataclass
class SemanticEntropyResult:
    semantic_entropy: float
    clusters: List[Cluster]
    cluster_probabilities: List[float]
    top_clusters_preview: List[Dict[str, Any]]
 
 
def compute_semantic_entropy(
    candidate_texts: Sequence[str],
    scorer: Optional[SemanticScorer] = None,
    sample_weights: Optional[Sequence[float]] = None,
) -> SemanticEntropyResult:
    clusters, assignment = cluster_semantics(candidate_texts, scorer=scorer)
    if not clusters:
        print("[CoherenceAnalyzer] No clusters found")
        return SemanticEntropyResult(semantic_entropy=0.0, clusters=[], cluster_probabilities=[], top_clusters_preview=[])
    p = estimate_cluster_probabilities(assignment, sample_weights=sample_weights)
    h = compute_entropy(p)
    # small preview for transparency
    previews: List[Dict[str, Any]] = []
    for c in clusters:
        example = candidate_texts[c.representative] if c.representative is not None else candidate_texts[c.members[0]]
        previews.append({
            "cluster_index": c.index,
            "num_members": len(c.members),
            "probability": p[c.index] if c.index < len(p) else 0.0,
            "representative_text": example[:280],
        })
    previews.sort(key=lambda d: d.get("probability", 0.0), reverse=True)
    return SemanticEntropyResult(semantic_entropy=h, clusters=clusters, cluster_probabilities=p, top_clusters_preview=previews)
 
 
def compute_conditional_semantic_entropy(
    prev_candidates: Sequence[str],
    next_candidates: Sequence[str],
    scorer: Optional[SemanticScorer] = None,
) -> float:
    """
    H(next | prev) estimated from clusters by conditioning on prev-clusters.
    Approximation: assume uniform within-cluster selection for prev.
    """
    if not prev_candidates or not next_candidates:
        return 0.0
    prev_res = compute_semantic_entropy(prev_candidates, scorer=scorer)
    next_res = compute_semantic_entropy(next_candidates, scorer=scorer)
    # Without joint samples, use weighted average of next entropy by prev cluster probs
    return float(sum(prev_res.cluster_probabilities) * next_res.semantic_entropy)
 
 
def compute_transitional_entailment(
    prev_response: str,
    next_response: str,
    scorer: Optional[SemanticScorer] = None,
) -> float:
    """
    A lightweight proxy for step-to-step coherence. If actions in `next` are
    logically consistent with `prev` (e.g., Navigate -> Pick same target),
    return high score. Falls back to scorer-based entailment between full texts.
    """
    prev_actions = parse_agent_actions_from_response(prev_response)
    next_actions = parse_agent_actions_from_response(next_response)
    if prev_actions and next_actions:
        # Heuristic: for any agent, if next step uses an action that typically
        # depends on prev (e.g., Pick after Navigate to same target), reward it.
        score_components: List[float] = []
        for uid, (act_next, arg_next) in next_actions.items():
            if uid in prev_actions:
                act_prev, arg_prev = prev_actions[uid]
                if act_prev.lower() == "navigate" and act_next.lower() in {"pick", "place", "open", "close"}:
                    if arg_prev and arg_next and normalize_text(arg_prev) in normalize_text(arg_next):
                        score_components.append(1.0)
                    else:
                        score_components.append(0.7)  # loosely consistent
                elif act_prev.lower() == "pick" and act_next.lower() == "place":
                    score_components.append(1.0 if arg_prev and arg_next and arg_prev in arg_next else 0.7)
                else:
                    score_components.append(0.6)  # neutral/unknown, mildly consistent
            else:
                score_components.append(0.5)  # different agent focus; neutral
        return float(np.mean(score_components)) if score_components else 0.5

    # Fallback: text-level entailment
    scorer = scorer or _default_semantic_scorer
    rel, s = scorer(prev_response, next_response)
    if rel in {SemanticRelation.EQUIVALENT, SemanticRelation.ENTAILS}:
        return max(0.6, s)
    if rel == SemanticRelation.CONTRADICTS:
        return 1.0 - s
    return 0.5
 
 
def compute_contradiction_rate(
    responses: Sequence[str],
    scorer: Optional[SemanticScorer] = None,
) -> float:
    """Pairwise contradiction ratio among a list of responses."""
    n = len(responses)
    if n <= 1:
        return 0.0
    scorer = scorer or _default_semantic_scorer
    total_pairs = 0
    contradictory = 0
    for i in range(n):
        for j in range(i + 1, n):
            rel, _ = scorer(responses[i], responses[j])
            total_pairs += 1
            if rel == SemanticRelation.CONTRADICTS:
                contradictory += 1
    return float(contradictory / total_pairs) if total_pairs > 0 else 0.0
 
 
 # =============================
 # Fact accuracy
 # =============================
 
@dataclass
class FactCheckResult:
    accuracy: float
    checks: List[Dict[str, Any]]
 
 
def evaluate_fact_accuracy(
    response: str,
    world_state: Optional[Dict[str, Any]] = None,
    world_accessors: Optional[Dict[str, Callable[..., Any]]] = None,
    prev_selected_response: Optional[str] = None,
) -> FactCheckResult:
    """
    Very lightweight rule-based fact checking over structured action lines
    using `world_state` or provided `world_accessors`.

    Examples of checks (best-effort heuristics):
    - If response issues `Pick[obj]`, verify obj is known in world_state
    - If `Navigate[target]`, verify target is a known object/furniture
    - If `Place[obj, relation, furniture, ...]`, verify furniture exists

    Returns an accuracy fraction and a list of per-check details.
    """
    actions = parse_agent_actions_from_response(response)
    if not actions:
        return FactCheckResult(accuracy=1.0, checks=[])

    checks: List[Dict[str, Any]] = []
    num_true = 0
    num_total = 0

    # Minimal object registry from world_state if present
    known_objects = set()
    known_furnitures = set()
    if world_state is not None:
        objs = world_state.get("object_positions", {})
        furn = world_state.get("furniture_positions", {})
        known_objects.update(list(objs.keys()))
        known_furnitures.update(list(furn.keys()))
    # Accessor helpers (optional)
    acc = world_accessors or {}
    is_known_object = acc.get("is_known_object", lambda name: (name in known_objects))
    is_known_furniture = acc.get("is_known_furniture", lambda name: (name in known_furnitures))
    object_parent = acc.get("object_parent", lambda name: None)
    distance_agent_to_target = acc.get("distance_agent_to_target", lambda uid, tgt: None)
    is_relation_valid = acc.get("is_relation_valid", lambda rel: rel in {"on", "within"})
    near_threshold: float = acc.get("near_threshold", 1.5)

    for uid, (act, raw_arg) in actions.items():
        act_l = act.lower()
        success = True
        reason = ""
        if raw_arg is None:
            # No-arg actions like Wait are trivially factual
            success = True
        else:
            # For multi-arg actions like Place[a, rel, f, ...], split conservatively
            args = [a.strip() for a in raw_arg.split(",")]
            if act_l in {"pick", "navigate", "open", "close"}:
                target = args[0] if args else None
                if target:
                    # Known entity check via accessors or fallback sets
                    success = bool(is_known_object(target) or is_known_furniture(target))
                    reason = "known target" if success else "unknown target"
                    # Optional proximity check for Pick
                    if act_l == "pick" and success:
                        d = distance_agent_to_target(uid, target)
                        if d is not None and d > near_threshold:
                            success = False
                            reason = f"too far to pick (d={d:.2f})"
            elif act_l == "place" or act_l == "rearrange":
                # Expect: obj, relation, furniture, ...
                obj = args[0] if len(args) >= 1 else None
                furniture = args[2] if len(args) >= 3 else None
                rel = args[1] if len(args) >= 2 else None
                # Check existence and relation validity
                if obj and not is_known_object(obj):
                    success = False
                    reason = "unknown obj"
                elif furniture and not is_known_furniture(furniture):
                    success = False
                    reason = "unknown furniture"
                elif rel and not is_relation_valid(rel):
                    success = False
                    reason = "invalid relation"
                # Optional spatial plausibility: near furniture before placing
                if success and furniture:
                    d = distance_agent_to_target(uid, furniture)
                    if d is not None and d > near_threshold:
                        # Not strictly false, but mark as likely issue
                        success = False
                        reason = f"too far to place (d={d:.2f})"
            else:
                # Unknown action types: do not penalize
                success = True

        checks.append({
            "agent": uid,
            "action": act,
            "args": raw_arg,
            "pass": bool(success),
            "reason": reason,
            "type": "entity/spatial",
        })
        num_total += 1
        num_true += 1 if success else 0

    # Temporal consistency checks using previous selected response (if provided)
    if prev_selected_response:
        prev_actions = parse_agent_actions_from_response(prev_selected_response)
        curr_actions = parse_agent_actions_from_response(response)
        for uid, (act_curr, arg_curr) in curr_actions.items():
            ok = True
            why = ""
            if uid in prev_actions:
                act_prev, arg_prev = prev_actions[uid]
                if act_curr.lower() == "pick":
                    # Prefer Navigate[target] -> Pick[target]
                    if act_prev.lower() == "navigate" and arg_prev and arg_curr and normalize_text(arg_prev) in normalize_text(arg_curr):
                        ok = True
                    else:
                        # Optional proximity check can justify
                        d = distance_agent_to_target(uid, arg_curr or "")
                        ok = (d is None) or (d <= near_threshold)
                        why = "no prior navigate" if d is None else ("near enough" if ok else f"far (d={d:.2f})")
                elif act_prev.lower() == "pick" and act_curr.lower() == "place":
                    ok = True
                elif act_curr.lower() == "navigate" and act_prev.lower() in {"pick", "place"}:
                    ok = True  # Backtracking allowed
            checks.append({
                "agent": uid,
                "action": act_curr,
                "args": arg_curr,
                "pass": bool(ok),
                "reason": why,
                "type": "temporal",
            })
            num_total += 1
            num_true += 1 if ok else 0

    accuracy = float(num_true / num_total) if num_total > 0 else 1.0
    return FactCheckResult(accuracy=accuracy, checks=checks)
 
 
 # =============================
 # Aggregate report
 # =============================
 
@dataclass
class CoherenceReport:
    # Core uncertainty/coherence
    step_semantic_entropy: float
    conditional_semantic_entropy: Optional[float]
    transitional_entailment: Optional[float]
    contradiction_rate: Optional[float]

    # Fact accuracy
    fact_accuracy: float

    # Transparency previews
    top_semantic_clusters: List[Dict[str, Any]]

    # Optional context echoes for traceability
    num_candidates: int
    selected_response_preview: str

    # Human-readable flags
    alerts: List[str] = field(default_factory=list)

    # Additional Transparency metrics (single-response friendly)
    # S_TL: STL/LTL-like one-step task-alignment score w.r.t. MIQP phase
    # S_TE: Transitional entailment score (alias of transitional_entailment)
    s_tl: Optional[float] = None
    s_te: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_semantic_entropy": self.step_semantic_entropy,
            "conditional_semantic_entropy": self.conditional_semantic_entropy,
            "transitional_entailment": self.transitional_entailment,
            "contradiction_rate": self.contradiction_rate,
            "fact_accuracy": self.fact_accuracy,
            "top_semantic_clusters": self.top_semantic_clusters,
            "num_candidates": self.num_candidates,
            "selected_response_preview": self.selected_response_preview,
            "alerts": self.alerts,
            "s_tl": self.s_tl,
            "s_te": self.s_te,
        }
 
 
class CoherenceAnalyzer:
    """
    High-level façade to compute semantic entropy, coherence and fact accuracy.
    """

    def __init__(
        self,
        semantic_scorer: Optional[SemanticScorer] = None,
        semantic_entropy_alert_threshold: float = 0.7,
        conditional_entropy_alert_threshold: float = 0.7,
        transitional_entailment_alert_threshold: float = 0.5,
        fact_accuracy_alert_threshold: float = 0.7,
    ) -> None:
        self.scorer = semantic_scorer or _default_semantic_scorer
        self.tau_sem = semantic_entropy_alert_threshold
        self.tau_csem = conditional_entropy_alert_threshold
        self.tau_ent = transitional_entailment_alert_threshold
        self.tau_fact = fact_accuracy_alert_threshold

    def generate_report(
        self,
        candidate_responses: Sequence[str],
        selected_response: str,
        prev_selected_response: Optional[str] = None,
        world_state: Optional[Dict[str, Any]] = None,
        world_accessors: Optional[Dict[str, Callable[..., Any]]] = None,
        sample_weights: Optional[Sequence[float]] = None,
        miqp_context: Optional[Dict[str, Any]] = None,
    ) -> CoherenceReport:
        # 1) Semantic entropy at current step
        sem_res = compute_semantic_entropy(candidate_responses or [selected_response], scorer=self.scorer, sample_weights=sample_weights)

        # 2) Conditional semantic entropy relative to prev selection
        cond_h: Optional[float] = None
        if prev_selected_response:
            cond_h = compute_conditional_semantic_entropy([prev_selected_response], candidate_responses or [selected_response], scorer=self.scorer)

        # 3) Transitional entailment score
        entail: Optional[float] = None
        if prev_selected_response:
            entail = compute_transitional_entailment(prev_selected_response, selected_response, scorer=self.scorer)

        # 4) Pairwise contradiction among candidates
        contra: Optional[float] = None
        if candidate_responses and len(candidate_responses) >= 2:
            contra = compute_contradiction_rate(candidate_responses, scorer=self.scorer)

        # 5) Fact accuracy
        fact = evaluate_fact_accuracy(
            selected_response,
            world_state=world_state,
            world_accessors=world_accessors,
            prev_selected_response=prev_selected_response,
        )

        # 6) STL/LTL-like one-step MIQP-aligned score (S_TL)
        s_tl: Optional[float] = None
        try:
            if miqp_context is not None:
                s_tl = compute_stl_like_task_alignment(
                    selected_response=selected_response,
                    current_phase=miqp_context.get("current_phase"),
                    alpha=miqp_context.get("alpha"),
                    aptitude_matrix=(
                        miqp_context.get("phase_task_info", {}).get("aptitude_matrix")
                        if isinstance(miqp_context.get("phase_task_info"), dict)
                        else None
                    ),
                )
        except Exception:
            s_tl = None

        # 7) Alerts
        alerts: List[str] = []
        if sem_res.semantic_entropy >= self.tau_sem:
            alerts.append("High semantic entropy: possible ambiguity or hallucination risk")
        if cond_h is not None and cond_h >= self.tau_csem:
            alerts.append("High conditional entropy: next step under-specified given previous")
        if entail is not None and entail < self.tau_ent:
            alerts.append("Low transitional entailment: potential coherence issue between steps")
        if fact.accuracy < self.tau_fact:
            alerts.append("Low fact accuracy: action arguments may not match known world state")

        return CoherenceReport(
            step_semantic_entropy=float(sem_res.semantic_entropy),
            conditional_semantic_entropy=cond_h,
            transitional_entailment=entail,
            contradiction_rate=contra,
            fact_accuracy=float(fact.accuracy),
            top_semantic_clusters=sem_res.top_clusters_preview,
            num_candidates=len(candidate_responses) if candidate_responses else 1,
            selected_response_preview=selected_response[:280],
            alerts=alerts,
            s_tl=s_tl,
            s_te=entail,
        )


# =============================
# STL/LTL-like one-step alignment score (S_TL)
# =============================

def _normalize_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return normalize_text(s)


def _task_matches_action(task: Dict[str, Any], action_name: str, action_arg: Optional[str]) -> bool:
    """
    Heuristic matching between a phase task dict and a parsed action line.
    Expected task keys: 'task_type', 'target' (strings). Optionally others.
    """
    try:
        ttype = _normalize_name(task.get("task_type"))
        tgt = _normalize_name(task.get("target"))
        aname = _normalize_name(action_name)
        aarg = _normalize_name(action_arg) if action_arg else None

        if ttype is None or aname is None:
            return False

        # Basic verbs mapping
        if ttype.startswith("navigate"):
            return aname == "navigate" and (tgt is None or (aarg and tgt in aarg))
        if ttype.startswith("pick"):
            return aname == "pick" and (tgt is None or (aarg and tgt in aarg))
        if ttype.startswith("place") or ttype.startswith("rearrange"):
            # place obj on/within furniture: allow partial match on object or furniture
            if aname != "place" and aname != "rearrange":
                return False
            if aarg is None:
                return tgt is None
            return (tgt is None) or (tgt in aarg)
        if ttype in {"open", "close"}:
            return aname == ttype and (tgt is None or (aarg and tgt in aarg))
        # Fallback: same verb
        return aname == ttype
    except Exception:
        return False


def compute_stl_like_task_alignment(
    selected_response: str,
    current_phase: Optional[Dict[str, Any]] = None,
    alpha: Optional[np.ndarray] = None,
    aptitude_matrix: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Compute S_TL: a single-step STL/LTL-like alignment score with the current MIQP phase.
    It rewards covering higher-weight tasks (weights derived from alpha or aptitude).

    Returns None if insufficient context.
    """
    if not current_phase or "tasks" not in current_phase:
        return None

    tasks: List[Dict[str, Any]] = current_phase.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) == 0:
        return None

    # Parse actions from the selected response
    actions = parse_agent_actions_from_response(selected_response)
    if not actions:
        # No actions means no coverage
        return 0.0

    # Derive per-task weights w_j
    num_tasks = len(tasks)
    weights = np.ones(num_tasks, dtype=float)
    if isinstance(alpha, np.ndarray) and alpha.ndim == 2 and alpha.shape[1] == num_tasks:
        weights = np.maximum(alpha, 0.0).sum(axis=0)
    elif isinstance(aptitude_matrix, np.ndarray) and aptitude_matrix.ndim == 2 and aptitude_matrix.shape[1] == num_tasks:
        weights = np.maximum(aptitude_matrix, 0.0).sum(axis=0)

    # Normalize weights
    wsum = float(weights.sum())
    if wsum > 0:
        weights = weights / wsum

    # Task coverage c_j
    coverage = np.zeros(num_tasks, dtype=float)
    for j, task in enumerate(tasks):
        covered = False
        for (_uid, (aname, aarg)) in actions.items():
            if _task_matches_action(task, aname, aarg):
                covered = True
                break
        coverage[j] = 1.0 if covered else 0.0

    score = float((coverage * weights).sum()) if num_tasks > 0 else 0.0
    return score
