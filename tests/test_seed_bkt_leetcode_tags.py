"""
tests/test_seed_bkt_leetcode_tags.py

Tests controllers/seeding_controller.py::get_lc_submissions' fix for the
"topicTags: []" bug -- LeetCode's recentSubmissionList doesn't return
topicTags, only question(titleSlug) does. get_lc_submissions must do a
second query per UNIQUE problem and attach translated (canonical ML)
tags in the same shape downstream code already expects.

Run:
    python -m pytest tests/test_seed_bkt_leetcode_tags.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from controllers.seeding_controller import get_lc_submissions, _fetch_question_topic_tags


def _resp(json_body, status=200):
    m = MagicMock()
    m.json.return_value = json_body
    m.raise_for_status.return_value = None
    return m


class TestLeetCodeTagResolution(unittest.TestCase):

    def test_fetch_question_topic_tags_translates_leetcode_slugs(self):
        with patch("controllers.seeding_controller.requests.post") as mock_post:
            mock_post.return_value = _resp({
                "data": {"question": {"topicTags": [{"slug": "hash-table"}, {"slug": "array"}]}}
            })
            tags = _fetch_question_topic_tags("two-sum")
        self.assertIn("hash_map", tags)   # hash-table -> hash_map via official_leetcode_tag_map
        self.assertIn("array", tags)

    def test_fetch_question_topic_tags_returns_empty_on_error(self):
        with patch("controllers.seeding_controller.requests.post") as mock_post:
            mock_post.return_value = _resp({"errors": [{"message": "not found"}]})
            tags = _fetch_question_topic_tags("nonexistent-problem")
        self.assertEqual(tags, [])

    def test_get_lc_submissions_attaches_tags_via_second_query(self):
        recent_list_response = _resp({
            "data": {
                "recentSubmissionList": [
                    {"title": "Two Sum", "titleSlug": "two-sum",
                     "statusDisplay": "Accepted", "timestamp": "1700000000"},
                ]
            }
        })
        question_response = _resp({
            "data": {"question": {"topicTags": [{"slug": "array"}, {"slug": "hash-table"}]}}
        })

        with patch("controllers.seeding_controller.requests.post") as mock_post:
            mock_post.side_effect = [recent_list_response, question_response]
            submissions = get_lc_submissions("some_handle")

        self.assertEqual(len(submissions), 1)
        tags = [t["slug"] for t in submissions[0]["topicTags"]]
        self.assertIn("array", tags)
        self.assertIn("hash_map", tags)

    def test_get_lc_submissions_only_queries_each_unique_problem_once(self):
        recent_list_response = _resp({
            "data": {
                "recentSubmissionList": [
                    {"title": "Two Sum", "titleSlug": "two-sum",
                     "statusDisplay": "Accepted", "timestamp": "1700000000"},
                    {"title": "Two Sum", "titleSlug": "two-sum",
                     "statusDisplay": "Wrong Answer", "timestamp": "1699999000"},
                ]
            }
        })
        question_response = _resp({
            "data": {"question": {"topicTags": [{"slug": "array"}]}}
        })

        with patch("controllers.seeding_controller.requests.post") as mock_post:
            mock_post.side_effect = [recent_list_response, question_response]
            submissions = get_lc_submissions("some_handle")

        # recentSubmissionList (1 call) + ONE question() call for the
        # single unique titleSlug shared by both submissions, not two.
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(len(submissions), 2)
        for sub in submissions:
            self.assertEqual([t["slug"] for t in sub["topicTags"]], ["array"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
