# db/fine_tune_hospital_insurance_generator.py
import json
import random
import os
from itertools import combinations
from settings.config import LOGGER, FINE_TUNE_DATA_PATH

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
# Instruction templates
# -----------------------------
HOSPITAL_QUERIES = [
    "What insurance plans are accepted at {hospital_name}?",
    "Which providers are linked to {hospital_name}?",
    "What specialties does {hospital_name} offer?",
    "Tell me about {hospital_name}'s services and coverage options."
]

COMPARISON_QUERIES = [
    "Compare {hospital_a} and {hospital_b}.",
    "Which hospital is better — {hospital_a} or {hospital_b}?",
    "Differences between {hospital_a} and {hospital_b}."
]

TERM_QUERIES = [
    "What is '{term}' in health insurance?",
    "Define '{term}'.",
    "Explain the meaning of '{term}'."
]

# -----------------------------
# Fine-Tuning Generator
# -----------------------------
# Conversational fine-tuning generator
def generate_insurance_fine_tuning_data_dialogue(hospitals, hospital_insurance_links, num_samples_per_hospital=15):
    fine_tuning_examples = []

    # Map hospital_id -> list of insurance plans
    hospital_to_plans = {}
    for link in hospital_insurance_links:
        hospital_to_plans.setdefault(link.hospital.hospital_id, []).append(link.insurance_plan)

    # Dialogue-style queries
    HOSPITAL_QUERIES = [
        "Which providers are linked to {hospital_name}?",
        "Tell me which insurance plans are accepted at {hospital_name}.",
        "Can you list the insurance options for {hospital_name}?",
        "What insurance coverage is available at {hospital_name}?"
    ]

    for hospital in hospitals:
        linked_plans = hospital_to_plans.get(hospital.hospital_id, [])
        plan_texts = [
            f"{p.plan_name} by {p.provider_name}" for p in linked_plans
        ]
        plan_sentence = ", ".join(plan_texts) if plan_texts else "no insurance plans currently available"
        
        for _ in range(num_samples_per_hospital):
            instr = random.choice(HOSPITAL_QUERIES).format(hospital_name=hospital.hospital_name)
            
            # Dialogue-style response
            response = (
                f"{hospital.hospital_name}, located in {hospital.location}, offers {hospital.hospital_type} services. "
                f"It has a rating of {hospital.rating}. "
                f"The hospital works with several insurance providers including {plan_sentence}. "
                f"Patients can check with the hospital for plan-specific benefits such as cashless facilities, coverage details, and network hospitals."
            )

            # Lean context
            context = {
                "hospital_id": hospital.hospital_id,
                "hospital_name": hospital.hospital_name,
                "location": hospital.location,
                "hospital_type": hospital.hospital_type,
                "insurance_providers": [link.insurance_plan.provider_name for link in hospital.insurance_plans_link],
                "rating": hospital.rating
            }

            fine_tuning_examples.append({
                "instruction": instr,
                "context": context,
                "response": response
            })

    # Add domain-specific terms for grounding
    for term, definition in DOMAIN_SPECIFIC_TERMS.items():
        instr = random.choice(TERM_QUERIES).format(term=term)
        fine_tuning_examples.append({
            "instruction": instr,
            "context": {"term": term},
            "response": definition
        })

    # Save JSONL
    os.makedirs(os.path.dirname(FINE_TUNE_DATA_PATH), exist_ok=True)
    with open(FINE_TUNE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_examples, f, indent=4)

    LOGGER.info(f"Generated {len(fine_tuning_examples)} conversational fine-tuning examples at {FINE_TUNE_DATA_PATH}")
    return fine_tuning_examples
