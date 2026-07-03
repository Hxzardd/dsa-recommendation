from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
submission = normalize(request)
outcome = run_rules(submission)

prompt = build_prompt(submission, outcome)

print("=" * 80)
print("SYSTEM PROMPT")
print("=" * 80)
print(prompt.system)

print()
print("=" * 80)
print("USER PROMPT")
print("=" * 80)
print(prompt.user)