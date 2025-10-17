# db/fine_tune_data_generator.py
# --------------------
# Generates fine-tuning data for the LLM based on hospital and insurance data.

import json
import random
from pony.orm import db_session, select
from db.db import Hospital, InsurancePlan, HospitalInsurancePlan, db
from db.modules.insurance_generator import get_domain_specific_terms # To get terms and definitions
from settings.config import LOGGER, FINE_TUNE_DATA_PATH, HOSPITAL_DB_FOLDER, HOSPITAL_DB_FILE_NAME
import os

# Ensure the database is bound
sqlite_db_path = os.path.abspath(os.path.join(HOSPITAL_DB_FOLDER, HOSPITAL_DB_FILE_NAME))
db.bind(provider='sqlite', filename=sqlite_db_path, create_db=True)
db.generate_mapping(create_tables=True)

def generate_fine_tuning_data(num_samples_per_type=20):
    """
    Generates a dataset for fine-tuning the LLM with insurance and hospital network data.
    The data will be in a JSONL-like format: [{"instruction": "...", "context": "...", "response": "..."}, ...]
    """
    fine_tuning_examples = []

    with db_session:
        hospitals = select(h for h in Hospital)[:]
        insurance_plans = select(ip for ip in InsurancePlan)[:]

        if not hospitals or not insurance_plans:
            LOGGER.warning("No hospitals or insurance plans found in DB. Cannot generate fine-tuning data.")
            return []

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
            plan = random.choice(insurance_plans)
            associated_hospitals = [
                hip.hospital for hip in HospitalInsurancePlan.select(lambda hip: hip.insurance_plan == plan)
            ]
            if not associated_hospitals:
                continue

            hospital = random.choice(associated_hospitals)
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
                "hospital_address": hospital.address
            }
            response = (
                f"Yes, {hospital.hospital_name} accepts the {plan.plan_name} plan from {plan.provider_name}. "
                f"It is part of their {plan.network_type} network."
            )
            fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})
        
        # 3. Generate examples for domain-specific terms
        domain_terms = get_domain_specific_terms()
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

if __name__ == "__main__":
    generate_fine_tuning_data()
