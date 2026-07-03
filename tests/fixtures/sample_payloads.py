# """Reusable request payload fixtures."""

VALID_WRONG_ANSWER_PAYLOAD = {
    "submission_id": "sub_5ac8f214",
    "problem_id": "binary_search_001",
    "user_id": "user_558",

    "language": "python",
    "verdict": "wrong_answer",

    "source_code": """def binary_search(nums, target):
    left = 0
    right = len(nums)

    while left < right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return -1
""",

    "test_summary": {
        "total_test_cases": 30,
        "passed_test_cases": 24,
        "failed_test_cases": 6,
    },

    "sample_failed_cases": [
        {
            "stdin": "nums=[1]\ntarget=1",
            "expected_output": "0",
            "actual_output": "-1",
        },
        {
            "stdin": "nums=[1,3,5,7]\ntarget=7",
            "expected_output": "3",
            "actual_output": "-1",
        },
        {
            "stdin": "nums=[2,4,6,8,10]\ntarget=10",
            "expected_output": "4",
            "actual_output": "-1",
        },
    ],

    "stdout": "-1",
    "stderr": "",
    "compile_output": "",

    "execution_time_ms": 39,
    "memory_kb": 14208,

    "submitted_at": "2026-07-02T10:15:43Z",
}


# VALID_RUNTIME_ERROR_PAYLOAD = {
#     "submission_id": "sub_runtime_001",
#     "problem_id": "two_sum_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "runtime_error",

#     "source_code": """def solve(nums):
#     for i in range(len(nums)):
#         print(nums[i + 1])
# """,

#     "test_summary": {
#         "total_test_cases": 15,
#         "passed_test_cases": 3,
#         "failed_test_cases": 12,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "1 2 3",
#             "expected_output": "1\n2\n3",
#             "actual_output": "",
#         }
#     ],

#     "stdout": "",
#     "stderr": "IndexError: list index out of range",
#     "compile_output": "",

#     "execution_time_ms": 2,
#     "memory_kb": 9120,

#     "submitted_at": "2026-07-04T09:00:00Z",
# }


# VALID_COMPILATION_ERROR_PAYLOAD = {
#     "submission_id": "sub_compile_001",
#     "problem_id": "sum_array_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "compilation_error",

#     "source_code": """def solve(nums)
#     return sum(nums)
# """,

#     "test_summary": {
#         "total_test_cases": 0,
#         "passed_test_cases": 0,
#         "failed_test_cases": 0,
#     },

#     "sample_failed_cases": [],

#     "stdout": "",
#     "stderr": "",
#     "compile_output": "SyntaxError: expected ':'",

#     "execution_time_ms": 0,
#     "memory_kb": 0,

#     "submitted_at": "2026-07-04T09:15:00Z",
# }


# VALID_TLE_PAYLOAD = {
#     "submission_id": "sub_tle_001",
#     "problem_id": "factorial_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "time_limit_exceeded",

#     "source_code": """def solve(n):
#     while True:
#         pass
# """,

#     "test_summary": {
#         "total_test_cases": 20,
#         "passed_test_cases": 5,
#         "failed_test_cases": 15,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "100000",
#             "expected_output": "933262154439...",
#             "actual_output": "",
#         }
#     ],

#     "stdout": "",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 2000,
#     "memory_kb": 11000,

#     "submitted_at": "2026-07-04T09:30:00Z",
# }

# VALID_ACCEPTED_PAYLOAD = {
#     "submission_id": "sub_accept_001",
#     "problem_id": "sum_array_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "accepted",

#     "source_code": """def solve(nums):
#     return sum(nums)
# """,

#     "test_summary": {
#         "total_test_cases": 20,
#         "passed_test_cases": 20,
#         "failed_test_cases": 0,
#     },

#     "sample_failed_cases": [],

#     "stdout": "15",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 8,
#     "memory_kb": 10240,

#     "submitted_at": "2026-07-04T09:45:00Z",
# }