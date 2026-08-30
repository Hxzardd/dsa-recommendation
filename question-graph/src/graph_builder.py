import json
from pathlib import Path

import networkx as nx


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

PROBLEM_FILE = DATA_DIR / "problem_nodes.json"
TOPIC_FILE = DATA_DIR / "topic_nodes.json"
PROBLEM_TOPIC_EDGE_FILE = DATA_DIR / "problem_topic_edges.json"
TOPIC_TOPIC_EDGE_FILE = DATA_DIR / "topic_topic_edges.json"

OUTPUT_GRAPH = BASE_DIR / "pcg_graph.graphml"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def add_attrs(graph, node_id, **attrs):
    graph.add_node(
        node_id,
        **{key: clean_value(value) for key, value in attrs.items()}
    )


def build_graph(problems, topics, problem_topic_edges, topic_topic_edges):
    graph = nx.DiGraph()
    unresolved_edges = []

    for p in problems:
        problem_id = p["problem_id"]

        add_attrs(
            graph,
            f"problem:{problem_id}",
            node_type="problem",
            problem_id=problem_id,
            title=p.get("title"),
            difficulty_score=p.get("difficulty_score"),
            cf_rating=p.get("cf_rating"),
            contest_id=p.get("contest_id"),
            problem_index=p.get("problem_index"),
            source=p.get("source"),
            url=p.get("url"),
            tags=p.get("tags"),
            skill_types=p.get("skill_types"),
            time_limit=p.get("time_limit"),
            memory_limit=p.get("memory_limit"),
            expected_time_complexity=p.get("expected_time_complexity"),
            expected_space_complexity=p.get("expected_space_complexity"),
        )

    for t in topics:
        topic_id = t["topic_id"]

        add_attrs(
            graph,
            f"topic:{topic_id}",
            node_type="topic",
            topic_id=topic_id,
            topic_name=t.get("topic_name"),
            cf_tag=t.get("cf_tag"),
            difficulty_level=t.get("difficulty_level"),
            is_root_topic=t.get("is_root_topic"),
        )

    for e in problem_topic_edges:
        problem_node = f"problem:{e['problem_id']}"
        topic_node = f"topic:{e['topic_id']}"

        missing_nodes = [node for node in (problem_node, topic_node) if node not in graph]
        if missing_nodes:
            unresolved_edges.append(
                f"problem-topic edge {e!r} references missing node(s): "
                f"{', '.join(missing_nodes)}"
            )
        else:
            graph.add_edge(
                problem_node,
                topic_node,
                edge_type="has_topic",
                is_primary_topic=clean_value(e.get("is_primary_topic")),
                # Topic relevance carried through from regenerate_graph_artifacts:
                # role='domain'|'optional', weight 0..1 (see that script).
                role=clean_value(e.get("role")),
                weight=clean_value(e.get("weight", 1.0)),
            )

    for e in topic_topic_edges:
        source_id = e.get("source_topic_id") or e.get("source")
        target_id = e.get("target_topic_id") or e.get("target")
        if not source_id or not target_id:
            raise KeyError(f"Topic edge {e!r} is missing source/target identifiers.")
        source_topic = f"topic:{source_id}"
        target_topic = f"topic:{target_id}"

        missing_nodes = [node for node in (source_topic, target_topic) if node not in graph]
        if missing_nodes:
            unresolved_edges.append(
                f"topic-topic edge {e!r} references missing node(s): "
                f"{', '.join(missing_nodes)}"
            )
        else:
            graph.add_edge(
                source_topic,
                target_topic,
                edge_type=e.get("relation_type", "prerequisite"),
                strength=clean_value(e.get("strength", 1.0)),
            )

    if unresolved_edges:
        details = "\n".join(f"- {error}" for error in unresolved_edges)
        raise ValueError(
            f"Cannot build graph: {len(unresolved_edges)} unresolved edge(s):\n{details}"
        )

    return graph


def main():
    problems = load_json(PROBLEM_FILE)
    topics = load_json(TOPIC_FILE)
    problem_topic_edges = load_json(PROBLEM_TOPIC_EDGE_FILE)
    topic_topic_edges = load_json(TOPIC_TOPIC_EDGE_FILE)

    graph = build_graph(problems, topics, problem_topic_edges, topic_topic_edges)

    print("Problems:", len(problems))
    print("Topics:", len(topics))
    print("Problem-topic edges:", len(problem_topic_edges))
    print("Topic-topic edges:", len(topic_topic_edges))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    nx.write_graphml(graph, OUTPUT_GRAPH)
    print(f"Saved graph -> {OUTPUT_GRAPH}")


if __name__ == "__main__":
    main()
