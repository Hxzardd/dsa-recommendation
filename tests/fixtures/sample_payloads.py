"""Reusable request payload fixtures."""

VALID_WRONG_ANSWER_PAYLOAD = {
    "submission_id": "sub_9f2a1b7c",
    "problem_id": "prob_1024",
    "user_id": "user_558",
    "language": "python",
    "verdict": "wrong_answer",
    "source_code": "def solve(nums, target):\n    ...",
    "stdin": "5\n1 2 3 4 5\n",
    "expected_output": "9\n",
    "actual_output": "8\n",
    "stdout": "8\n",
    "stderr": "",
    "compile_output": "",
    "execution_time_ms": 42,
    "memory_kb": 15360,
    "submitted_at": "2026-06-30T18:04:11Z",
}

