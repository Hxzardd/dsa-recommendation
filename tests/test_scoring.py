"""Unit tests for the ML-owned submission score formulation (scoring.py).

Kept in weight-parity with the backend's src/server/scoring.ts::calculateScore.
"""

import unittest

from pipeline.recommender.scoring import compute_submission_score


def _submission(**telemetry):
    base = {
        "verdict": "OK",
        "submissionCount": 1,
        "hintsUsed": 0,
        "testCasesPassed": 10,
        "totalTestCases": 10,
        "problemTopics": [],
    }
    if telemetry:
        base["telemetry"] = telemetry
    return base


class TestSubmissionScore(unittest.TestCase):
    def test_missing_telemetry_is_neutral_not_penalised(self):
        out = compute_submission_score(_submission())
        # No telemetry -> neutral components, still a real score.
        self.assertGreater(out["finalScore"], 0.0)
        self.assertEqual(out["fluencyScore"], 0.5)
        self.assertEqual(out["speedScore"], 0.5)

    def test_score_and_normalised_agree(self):
        out = compute_submission_score(
            _submission(
                majorRewriteCount=0,
                backspaceCount=1,
                totalKeystrokes=100,
                sessionDurationSeconds=90,
                edgeCasesPassed=10,
                totalEdgeCases=10,
                runtimePercentile=0.1,
                memoryPercentile=0.2,
            )
        )
        self.assertEqual(out["normalisedScore"], round(out["finalScore"] * 100))
        self.assertGreaterEqual(out["finalScore"], 0.0)
        self.assertLessEqual(out["finalScore"], 1.0)

    def test_first_pass_rewards_clean_first_solve(self):
        clean = compute_submission_score(_submission())
        retried = compute_submission_score(
            {**_submission(), "submissionCount": 5}
        )
        self.assertEqual(clean["firstPass"], 1.0)
        self.assertEqual(retried["firstPass"], 0.0)
        self.assertGreater(clean["rawScore"], retried["rawScore"])

    def test_hint_use_penalises(self):
        no_hint = compute_submission_score(
            _submission(hintsUsed=0, sessionDurationSeconds=100)
        )
        early_hint = compute_submission_score(
            _submission(
                hintsUsed=3,
                firstHintOpenedAtSeconds=5,
                sessionDurationSeconds=100,
            )
        )
        self.assertGreater(early_hint["hintPenalty"], 0.0)
        self.assertGreater(no_hint["rawScore"], early_hint["rawScore"])

    def test_previous_score_smoothing_uses_prior_mastery(self):
        sub = _submission()
        sub["problemTopics"] = [
            {"topicId": "array", "currentMastery": 0.8},
            {"topicId": "hash_map", "currentMastery": 0.6},
        ]
        out = compute_submission_score(sub)
        self.assertAlmostEqual(out["previousScore"], 0.7, places=4)


if __name__ == "__main__":
    unittest.main()
