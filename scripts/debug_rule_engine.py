from pprint import pprint

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
submission = normalize(request)

outcome = run_rules(submission)

print("=" * 80)
print("RULE ENGINE")
print("=" * 80)

pprint(outcome.model_dump())