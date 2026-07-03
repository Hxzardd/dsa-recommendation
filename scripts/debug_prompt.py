from app.config.settings import get_settings
from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

payload = VALID_WRONG_ANSWER_PAYLOAD.copy()

payload["source_code"] = (
    "def solve():\n"
    + "    x = 1\n" * 1500
)

request = AnalyzeRequest.model_validate(payload)
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

print()

print("=" * 80)
print("PROMPT LENGTHS")
print("=" * 80)

print("System Prompt:", len(prompt.system))
print("User Prompt  :", len(prompt.user))
print("Total        :", len(prompt.system) + len(prompt.user))

print("Original source length:", len(payload["source_code"]))
print("Prompt user length:", len(prompt.user))

settings = get_settings()
print("Prompt limit:", settings.prompt_max_chars)