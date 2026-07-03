"""Debug script for verifying parser normalization."""

from pprint import pprint

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def main() -> None:
    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)

    submission = normalize(request)

    print("=" * 80)
    print("RAW REQUEST")
    print("=" * 80)
    pprint(request.model_dump())

    print()

    print("=" * 80)
    print("NORMALIZED SUBMISSION")
    print("=" * 80)
    pprint(submission.model_dump())


if __name__ == "__main__":
    main()