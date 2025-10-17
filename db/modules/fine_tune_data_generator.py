# db/fine_tune_data_generator.py
# --------------------
# Generates fine-tuning data for the LLM based on hospital and insurance data.

import json
import random
import os
from settings.config import LOGGER, FINE_TUNE_DATA_PATH
from itertools import combinations

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

def generate_fine_tuning_data(hospitals, insurance_plans, hospital_insurance_plans, num_samples_per_type=50):
    """
    Generates a dataset for fine-tuning the LLM with insurance and hospital network data.
    Includes combinatorial augmentation for multiple hospitals, multiple insurance providers,
    and specialty-based filtering.
    Returns the file path where the JSONL dataset is saved.
    """
    fine_tuning_examples = []
    if not hospitals or not insurance_plans or not hospital_insurance_plans:
        LOGGER.warning("No hospitals, insurance plans, or hospital-insurance links found in DB. Cannot generate fine-tuning data.")
        return []

    # Create lookup dictionaries
    hospital_map = {h.hospital_id: h for h in hospitals}
    insurance_map = {p.plan_id: p for p in insurance_plans}
    hospital_to_insurance = {}
    insurance_to_hospitals = {}

    for link in hospital_insurance_plans:
        hospital_to_insurance.setdefault(link.hospital.hospital_id, []).append(link.insurance_plan.plan_id)
        insurance_to_hospitals.setdefault(link.insurance_plan.plan_id, []).append(link.hospital.hospital_id)

    # -----------------------------
    # 1. Single Hospital + Single Insurance (existing logic)
    # -----------------------------
    for _ in range(num_samples_per_type):
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
        response = f"Yes, {hospital.hospital_name} accepts the {plan.plan_name} plan from {plan.provider_name}. It is part of their {plan.network_type} network."
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # -----------------------------
    # 2. Hospital + Multiple Insurance Providers
    # -----------------------------
    for hospital_id, plan_ids in hospital_to_insurance.items():
        hospital = hospital_map[hospital_id]
        # Pick 2-3 random insurance plans linked to this hospital
        selected_plan_ids = random.sample(plan_ids, min(len(plan_ids), random.randint(2, 3)))
        selected_plans = [insurance_map[p_id] for p_id in selected_plan_ids]

        if not selected_plans:
            continue

        instruction = f"Which insurance plans are accepted at {hospital.hospital_name}?"
        context = {
            "hospital_name": hospital.hospital_name,
            "insurance_plan_names": [p.plan_name for p in selected_plans],
            "insurance_provider_names": [p.provider_name for p in selected_plans],
            "network_types": [p.network_type for p in selected_plans],
            "hospital_address": hospital.address
        }
        response_lines = [
            f"{hospital.hospital_name} accepts the {p.plan_name} plan from {p.provider_name} ({p.network_type} network)."
            for p in selected_plans
        ]
        response = " ".join(response_lines)
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # -----------------------------
    # 3. Multiple Hospitals for One Insurance Plan
    # -----------------------------
    for plan_id, hospital_ids in insurance_to_hospitals.items():
        plan = insurance_map[plan_id]
        selected_hospital_ids = random.sample(hospital_ids, min(len(hospital_ids), random.randint(2, 3)))
        selected_hospitals = [hospital_map[h_id] for h_id in selected_hospital_ids]

        instruction = f"Which hospitals accept the {plan.plan_name} insurance plan?"
        context = {
            "insurance_plan_name": plan.plan_name,
            "insurance_provider_name": plan.provider_name,
            "network_type": plan.network_type,
            "hospitals": [h.hospital_name for h in selected_hospitals]
        }
        response_lines = [
            f"{h.hospital_name} accepts the {plan.plan_name} plan from {plan.provider_name} ({plan.network_type} network)."
            for h in selected_hospitals
        ]
        response = " ".join(response_lines)
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # -----------------------------
    # 4. Specialty-Based Filtering Examples
    # -----------------------------
    specialties = list({h.hospital_type for h in hospitals})
    for _ in range(num_samples_per_type):
        specialty = random.choice(specialties)
        relevant_hospitals = [h for h in hospitals if h.hospital_type == specialty]
        if not relevant_hospitals:
            continue
        selected_hospitals = random.sample(relevant_hospitals, min(len(relevant_hospitals), random.randint(1, 3)))
        selected_plans = []
        for h in selected_hospitals:
            plan_ids = hospital_to_insurance.get(h.hospital_id, [])
            selected_plans.extend([insurance_map[p_id] for p_id in plan_ids])
        selected_plans = list({p.plan_id: p for p in selected_plans}.values())  # remove duplicates

        instruction = f"Which {specialty} hospitals accept insurance?"
        context = {
            "specialty": specialty,
            "hospitals": [h.hospital_name for h in selected_hospitals],
            "insurance_plan_names": [p.plan_name for p in selected_plans]
        }
        response_lines = [
            f"{h.hospital_name} ({specialty}) accepts insurance plans: {', '.join([p.plan_name for p in selected_plans])}."
            for h in selected_hospitals
        ]
        response = " ".join(response_lines)
        fine_tuning_examples.append({"instruction": instruction, "context": context, "response": response})

    # -----------------------------
    # 5. Domain-Specific Terms
    # -----------------------------
    for term, definition in DOMAIN_SPECIFIC_TERMS.items():
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

    # -----------------------------
    # Save to file
    # -----------------------------
    random.shuffle(fine_tuning_examples)
    output_dir = os.path.dirname(FINE_TUNE_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(FINE_TUNE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_examples, f, indent=4)
    
    LOGGER.info(f"Generated {len(fine_tuning_examples)} fine-tuning examples to {FINE_TUNE_DATA_PATH}")
    return FINE_TUNE_DATA_PATH
