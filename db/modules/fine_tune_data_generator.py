# db/fine_tune_data_generator.py
# --------------------
# Generates fine-tuning data for the LLM based on hospital and insurance data.

import json
import random
from settings.config import LOGGER, FINE_TUNE_DATA_PATH
import os

# Domain-Specific Terms for Fine-Tuning (UAE Healthcare Context)
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

def generate_fine_tuning_data(hospitals, insurance_plans, hospital_insurance_plans, num_samples_per_type=20):
    """
    Generates a dataset for fine-tuning the LLM with insurance and hospital network data.
    The data will be in a JSONL-like format: [{"instruction": "...", "context": "...", "response": "..."}, ...]
    """
    fine_tuning_examples = []
    if not hospitals or not insurance_plans or not hospital_insurance_plans:
        LOGGER.warning("No hospitals, insurance plans, or hospital-insurance links found in DB. Cannot generate fine-tuning data.")
        return []

    # Create dictionaries for quick lookup using Pony ORM entity primary keys
    hospital_map = {h.hospital_id: h for h in hospitals}
    insurance_map = {p.plan_id: p for p in insurance_plans}

    # 1. Generate examples for insurance plan details
    for _ in range(num_samples_per_type):
        plan = random.choice(insurance_plans)
        instruction_templates = [
            f"What are the policy terms for {plan.plan_name}?",
            f"Tell me about the coverage details of the {plan.plan_name} plan from {plan.provider_name}.",
            f"Explain the {plan.network_type} network type under {plan.plan_name}.",
            f"What does {plan.plan_name} by {plan.provider_name} cover?"
        ]
        instruction = random.choice(instruction_templates)
        context = {
            "plan_name": plan.plan_name,
            "provider_name": plan.provider_name,
            "policy_terms": plan.policy_terms,
            "coverage_details": plan.coverage_details,
            "network_type": plan.network_type
        }
        response = (
            f"The {plan.plan_name} plan from {plan.provider_name} has the following policy terms: "
            f"{plan.policy_terms}. Its coverage details include: {plan.coverage_details}. "
            f"It operates under a {plan.network_type} network."
        )
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # 2. Generate examples for hospital network information related to insurance
    for _ in range(num_samples_per_type):
        # Select a random hospital-insurance link
        link = random.choice(hospital_insurance_plans)
        
        hospital = hospital_map.get(link.hospital.hospital_id)
        plan = insurance_map.get(link.insurance_plan.plan_id)

        if not hospital or not plan:
            continue

        instruction_templates = [
            f"Does {hospital.hospital_name} accept {plan.plan_name} insurance?",
            f"Is {hospital.hospital_name} in the network for {plan.provider_name}?",
            f"Which insurance plans does {hospital.hospital_name} accept from {plan.provider_name}?",
            f"Can I use my {plan.plan_name} plan at {hospital.hospital_name}?"
        ]
        instruction = random.choice(instruction_templates)
        context = {
            "hospital_name": hospital.hospital_name,
            "insurance_plan_name": plan.plan_name,
            "insurance_provider_name": plan.provider_name,
            "network_type": plan.network_type,
            "hospital_address": hospital.address,
        }
        response = (
            f"Yes, {hospital.hospital_name} accepts the {plan.plan_name} plan from {plan.provider_name}. "
            f"It is part of their {plan.network_type} network."
        )
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})
    
    # 3. Generate examples for domain-specific terms
    domain_terms = DOMAIN_SPECIFIC_TERMS
    for term, definition in domain_terms.items():
        instruction_templates = [
            f"What is '{term}' in insurance?",
            f"Can you define '{term}'?",
            f"What does '{term}' mean?",
            f"Explain '{term}'."
        ]
        instruction = random.choice(instruction_templates)
        context = {"term": term, "definition": definition}
        response = definition
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # Shuffle and save to file
    random.shuffle(fine_tuning_examples)
    output_dir = os.path.dirname(FINE_TUNE_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(FINE_TUNE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_examples, f, indent=4)
    
    LOGGER.info(f"Generated {len(fine_tuning_examples)} fine-tuning examples to {FINE_TUNE_DATA_PATH}")
    return FINE_TUNE_DATA_PATH
