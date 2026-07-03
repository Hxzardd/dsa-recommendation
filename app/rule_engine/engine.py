"""Rule engine orchestration."""

from app.config.settings import get_settings
from app.models.domain import NormalizedSubmission, RuleEngineOutcome, RuleResult
from app.rule_engine.rules import REGISTERED_RULES


def _empty_outcome(note: str) -> RuleEngineOutcome:
    """Return an LLM-required outcome with a single trace note."""

    return RuleEngineOutcome(
        needs_llm=True,
        error_category=None,
        deterministic_feedback=None,
        deterministic_hint=None,
        rule_notes=[note],
    )


def _should_short_circuit(rule_result: RuleResult) -> bool:
    """Return whether one rule result is sufficient without an LLM."""

    return (
        rule_result.confidence == "high"
        and rule_result.deterministic_feedback is not None
        and rule_result.deterministic_hint is not None
    )


def run_rules(sub: NormalizedSubmission) -> RuleEngineOutcome:
    """Run deterministic rules in fixed priority order."""

    if not get_settings().rule_engine_enabled:
        return _empty_outcome("rule_engine: disabled by configuration")

    matched_results = []
    for rule in REGISTERED_RULES:
        rule_result = rule(sub)
        if rule_result is None or not rule_result.matched:
            continue

        matched_results.append(rule_result)
        if _should_short_circuit(rule_result):
            sub.is_deterministic_case = True
            return RuleEngineOutcome(
                needs_llm=False,
                error_category=rule_result.error_category,
                deterministic_feedback=rule_result.deterministic_feedback,
                deterministic_hint=rule_result.deterministic_hint,
                rule_notes=[result.note for result in matched_results],
            )

    if not matched_results:
        return _empty_outcome("rule_engine: no deterministic rule matched")

    rule_notes = [result.note for result in matched_results]
    category_result = next(
        (
            result
            for result in matched_results
            if result.error_category is not None and result.confidence in {"high", "medium"}
        ),
        None,
    )
    hint_result = next(
        (result for result in matched_results if result.deterministic_hint is not None),
        None,
    )

    return RuleEngineOutcome(
        needs_llm=True,
        error_category=category_result.error_category if category_result else None,
        deterministic_feedback=None,
        deterministic_hint=hint_result.deterministic_hint if hint_result else None,
        rule_notes=rule_notes,
    )
