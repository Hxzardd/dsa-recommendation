import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "src" / "graph_builder.py"
SPEC = importlib.util.spec_from_file_location("graph_builder", MODULE_PATH)
graph_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(graph_builder)


class BuildGraphTests(unittest.TestCase):
    def setUp(self):
        self.problems = [{"problem_id": "p1", "title": "Example"}]
        self.topics = [
            {"topic_id": "arrays", "topic_name": "Arrays"},
            {"topic_id": "sorting", "topic_name": "Sorting"},
        ]

    def test_builds_edges_when_all_endpoints_exist(self):
        graph = graph_builder.build_graph(
            self.problems,
            self.topics,
            [{"problem_id": "p1", "topic_id": "arrays"}],
            [{"source_topic_id": "arrays", "target_topic_id": "sorting"}],
        )

        self.assertTrue(graph.has_edge("problem:p1", "topic:arrays"))
        self.assertTrue(graph.has_edge("topic:arrays", "topic:sorting"))

    def test_builds_edges_with_source_target_format(self):
        graph = graph_builder.build_graph(
            self.problems,
            self.topics,
            [{"problem_id": "p1", "topic_id": "arrays"}],
            [{"source": "arrays", "target": "sorting"}],
        )

        self.assertTrue(graph.has_edge("problem:p1", "topic:arrays"))
        self.assertTrue(graph.has_edge("topic:arrays", "topic:sorting"))

    def test_reports_all_unresolved_edge_endpoints(self):
        with self.assertRaises(ValueError) as context:
            graph_builder.build_graph(
                self.problems,
                self.topics,
                [{"problem_id": "missing", "topic_id": "unknown"}],
                [{"source_topic_id": "arrays", "target_topic_id": "missing"}],
            )

        message = str(context.exception)
        self.assertIn("2 unresolved edge(s)", message)
        self.assertIn("problem:missing", message)
        self.assertIn("topic:unknown", message)
        self.assertIn("topic:missing", message)


if __name__ == "__main__":
    unittest.main()
