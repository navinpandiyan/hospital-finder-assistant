# db/fine_tune_insurance_generator.py
import json
import random
import os
from itertools import combinations
from settings.config import LOGGER, FINE_TUNE_DATA_PATH

# -----------------------------
# Instruction templates
# -----------------------------
POLICY_QUERIES = [
    "Explain the policy terms of {plan_name}.",
    "What are the terms and conditions of {plan_name} from {provider}?",
    "List the important clauses of {plan_name}.",
    "Summarize what {plan_name} covers under policy terms."
]

COVERAGE_QUERIES = [
    "Explain the coverage benefits of {plan_name} from {provider}.",
    "What does {plan_name} cover?",
    "List the coverage benefits under {plan_name}.",
    "Give me a summary of treatments covered under {plan_name}."
]

COMPARISON_QUERIES = [
    "Compare {plan_a} and {plan_b}.",
    "How does {plan_a} differ from {plan_b}?",
    "Which plan is better — {plan_a} or {plan_b}?",
    "What’s the difference between {plan_a} and {plan_b}?"
]

TERM_QUERIES = [
    "What is '{term}' in health insurance?",
    "Define '{term}'.",
    "Explain the meaning of '{term}'."
]

# -----------------------------
# Domain-Specific Terms
# -----------------------------
DOMAIN_SPECIFIC_TERMS = {
    "direct billing": "When the hospital directly settles the claim with the insurance provider without patient payment.",
    "pre-approval required": "Authorization from the insurer is needed before certain treatments or hospital admissions.",
    "gold plan network": "A premium network of hospitals offering wider benefits under the Gold insurance tier.",
    "cashless facility": "A feature where the insured receives treatment at a network hospital without paying upfront.",
    "co-payment": "The percentage of the medical bill that the insured person must pay out-of-pocket.",
    "waiting period": "A defined duration during which no claims are admissible for specific conditions.",
    "sum insured": "The maximum coverage amount payable by the insurer per policy year.",
    "reimbursement": "The process of refunding expenses when treatment is received outside the insurance network.",
    "policy exclusion": "Specific conditions or treatments not covered under the insurance plan.",
    "network hospital": "A hospital that has an active tie-up with an insurance provider for cashless or direct billing services."
}

# -----------------------------
# Fine-Tuning Data Generator
# -----------------------------
def generate_insurance_fine_tuning_data(insurance_plans: list, num_samples_per_plan: int = 5):
    """
    Generates fine-tuning JSONL data using a list of InsurancePlan objects.
    """
    fine_tuning_examples = []

    # -----------------------------
    # 1. Policy & Coverage per plan
    # -----------------------------
    for plan in insurance_plans:
        plan_name = plan.plan_name
        provider = plan.provider_name
        policy_terms = getattr(plan, "policy_terms", "")
        coverage_details = getattr(plan, "coverage_details", "")
        network_type = getattr(plan, "network_type", "")

        for _ in range(num_samples_per_plan):
            # Policy instruction
            instr = random.choice(POLICY_QUERIES).format(plan_name=plan_name, provider=provider)
            response_parts = []
            if policy_terms:
                response_parts.append(f"The {plan_name} plan by {provider} includes: {policy_terms}.")
            if network_type:
                response_parts.append(f"Network type: {network_type}.")
            if coverage_details:
                response_parts.append(f"Coverage details: {coverage_details}.")
            response = " ".join(response_parts)
            context = {
                "plan_name": plan_name,
                "provider": provider,
                "policy_terms": policy_terms,
                "coverage": coverage_details,
                "network_type": network_type
            }
            fine_tuning_examples.append({
                "instruction": instr,
                "context": context,
                "response": response
            })

            # Coverage instruction
            instr_cov = random.choice(COVERAGE_QUERIES).format(plan_name=plan_name, provider=provider)
            response_cov = f"{plan_name} by {provider} covers: {coverage_details or 'N/A'}. Network type: {network_type or 'N/A'}."
            fine_tuning_examples.append({
                "instruction": instr_cov,
                "context": context,
                "response": response_cov
            })

    # -----------------------------
    # 2. Comparative examples
    # -----------------------------
    plan_combinations = list(combinations(insurance_plans, 2))
    for plan_a, plan_b in random.sample(plan_combinations, min(len(plan_combinations), num_samples_per_plan)):
        instr_cmp = random.choice(COMPARISON_QUERIES).format(plan_a=plan_a.plan_name, plan_b=plan_b.plan_name)
        response_cmp = (
            f"{plan_a.plan_name} (by {plan_a.provider_name}) covers: {getattr(plan_a, 'coverage_details', 'N/A')}. "
            f"{plan_a.plan_name} follows {getattr(plan_a, 'network_type', 'N/A')} network. "
            f"In contrast, {plan_b.plan_name} (by {plan_b.provider_name}) covers: {getattr(plan_b, 'coverage_details', 'N/A')}. "
            f"{plan_b.plan_name} follows {getattr(plan_b, 'network_type', 'N/A')} network."
        )
        fine_tuning_examples.append({
            "instruction": instr_cmp,
            "context": {
                "plan_a": plan_a.plan_name,
                "plan_b": plan_b.plan_name,
                "provider_a": plan_a.provider_name,
                "provider_b": plan_b.provider_name
            },
            "response": response_cmp
        })

    # -----------------------------
    # 3. Domain-specific term definitions
    # -----------------------------
    for term, definition in DOMAIN_SPECIFIC_TERMS.items():
        instr_term = random.choice(TERM_QUERIES).format(term=term)
        fine_tuning_examples.append({
            "instruction": instr_term,
            "context": {"term": term},
            "response": definition
        })

    # -----------------------------
    # Save JSONL
    # -----------------------------
    os.makedirs(os.path.dirname(FINE_TUNE_DATA_PATH), exist_ok=True)
    with open(FINE_TUNE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_examples, f, indent=4)

    LOGGER.info(f"Generated {len(fine_tuning_examples)} insurance-focused fine-tuning examples at {FINE_TUNE_DATA_PATH}")
    return fine_tuning_examples

