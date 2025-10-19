# db/modules/fine_tune_data_generator.py
import asyncio
import json
import os
import random
from settings.config import LOGGER, FINE_TUNE_DATA_PATH
from tools.text_to_speech import text_to_dialogue

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

HOSPITAL_QUERIES = [
    "Which providers are linked to {hospital_name}?",
    "Tell me which insurance plans are accepted at {hospital_name}.",
    "Can you list the insurance options for {hospital_name}?",
    "What insurance coverage is available at {hospital_name}?"
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
def generate_insurance_fine_tuning_data_dialogue(hospitals, hospital_insurance_links, num_samples_per_hospital=3):
    fine_tuning_examples = []

    # Map hospital_id -> list of linked insurance plans
    hospital_to_plans = {}
    for link in hospital_insurance_links:
        hospital_to_plans.setdefault(link.hospital.hospital_id, []).append(link.insurance_plan)

    for hospital in hospitals:
        linked_plans = hospital_to_plans.get(hospital.hospital_id, [])

        for _ in range(num_samples_per_hospital):
            # Choose instruction
            instr = random.choice(HOSPITAL_QUERIES).format(hospital_name=hospital.hospital_name)

            # Select 1–2 domain-specific terms for grounding
            selected_terms = random.sample(list(DOMAIN_SPECIFIC_TERMS.keys()), k=random.randint(1, 2))
            terms_text = "; ".join([f"'{t}': {DOMAIN_SPECIFIC_TERMS[t]}" for t in selected_terms])

            # Construct Insurance Details section from linked plans
            insurance_details_lines = []
            for p in linked_plans:
                line = (
                    f"Plan Name: {p.plan_name}\n"
                    f"Provider: {p.provider_name}\n"
                    f"Policy Terms: {p.policy_terms}\n"
                    f"Coverage Details: {p.coverage_details}\n"
                    f"Network Type: {p.network_type}\n"
                    f"Rating: {p.rating}\n"
                )
                insurance_details_lines.append(line)
            insurance_details_text = "\n---\n".join(insurance_details_lines) if insurance_details_lines else "No insurance plans available."

            # Construct response with Hospital Details + Insurance Details
            response = (
                f"Hospital Details:\n"
                f"Name: {hospital.hospital_name}\n"
                f"Location: {hospital.location}\n"
                f"Services: {hospital.hospital_type}\n"
                f"Rating: {hospital.rating}\n\n"
                f"Insurance Details:\n{insurance_details_text}\n\n"
                f"Additional Info: {terms_text}"
            )

            # Context for reference
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

    # Add domain-specific terms as separate instructions
    for term, definition in DOMAIN_SPECIFIC_TERMS.items():
        instr = random.choice(TERM_QUERIES).format(term=term)
        fine_tuning_examples.append({
            "instruction": instr,
            "context": {"term": term},
            "response": definition
        })

    # Async function to convert all responses to dialogue
    async def convert_all_to_dialogue(fine_tuning_examples):
        async def process_item(item):
            dialogue_result = await text_to_dialogue(item["response"])
            item["response"] = dialogue_result.dialogue
            return item

        tasks = [process_item(item) for item in fine_tuning_examples]
        updated_examples = await asyncio.gather(*tasks)
        return updated_examples

    # Run async conversion
    fine_tuning_examples = asyncio.run(convert_all_to_dialogue(fine_tuning_examples))

    # Save JSONL
    os.makedirs(os.path.dirname(FINE_TUNE_DATA_PATH), exist_ok=True)
    with open(FINE_TUNE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_examples, f, indent=4)

    LOGGER.info(f"Converted all examples to dialogue and saved {len(fine_tuning_examples)} items at {FINE_TUNE_DATA_PATH}")
