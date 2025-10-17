# db/insurance_generator.py
# --------------------
# Generates synthetic insurance plan data.

import random
from settings.config import INSURANCE_PROVIDERS

# Common Insurance Plan Names
PLAN_NAMES = [
    "Gold Comprehensive", "Silver Select", "Bronze Basic", "Platinum Plus",
    "Family Floater", "Individual Protector", "Critical Illness Cover", "Corporate Advantage", "Smart Saver"
]

# Policy Terms and Clauses (with placeholders for random insertion)
POLICY_TERM_PHRASES = [
    "Covers {percentage}% of hospitalization costs.",
    "Pre-approval required for planned surgeries and major procedures.",
    "Direct billing available at listed network hospitals.",
    "30-day waiting period for general illnesses.",
    "Cashless facility available for network hospitals.",
    "Reimbursement applicable for non-network hospitals within 7 working days.",
    "Includes maternity benefits after a 9-month waiting period.",
    "Organ transplant coverage up to AED {amount}.",
    "Critical illness protection for up to 10 major conditions.",
    "No waiting period for accidental emergencies.",
    "No coverage for pre-existing conditions in the first year.",
    "Sum insured limited to AED {amount}.",
    "Co-payment of {copay}% applicable on outpatient visits."
]

# Additional Coverage Details
COVERAGE_DETAIL_PHRASES = [
    "Basic inpatient hospitalization and surgical coverage.",
    "Outpatient department (OPD) consultations covered up to AED {limit_amount} per year.",
    "Room rent capped at {room_rent_percentage}% of the total sum insured.",
    "Ambulance service covered up to AED {ambulance_amount} per claim.",
    "Domiciliary hospitalization included under special cases.",
    "Daycare procedures such as cataract and dialysis included.",
    "Specific disease coverage for conditions like {disease_list}.",
    "Annual preventive health check-up included.",
    "International emergency coverage available within GCC countries.",
    "No claim bonus added for claim-free years."
]

# Network Coverage Models
NETWORK_TYPES = [
    "PPO (Preferred Provider Organization)",
    "HMO (Health Maintenance Organization)",
    "EPO (Exclusive Provider Organization)",
    "POS (Point of Service)",
    "OAN (Open Access Network)"
]

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


def generate_insurance_plans(num_plans=50):
    """
    Generates a list of synthetic insurance plan records.
    """
    records = []
    for i in range(1, num_plans + 1):
        plan_name = random.choice(PLAN_NAMES)
        provider_name = random.choice(INSURANCE_PROVIDERS)
        network_type = random.choice(NETWORK_TYPES)

        # Generate policy terms
        policy_terms_parts = random.sample(POLICY_TERM_PHRASES, k=random.randint(2, 4))
        policy_terms = " ".join([
            part.format(
                percentage=random.randint(60, 100),
                amount=f"${random.randint(50000, 200000):,}",
                copay=random.randint(5, 20)
            ) for part in policy_terms_parts
        ])

        # Generate coverage details
        coverage_details_parts = random.sample(COVERAGE_DETAIL_PHRASES, k=random.randint(2, 3))
        coverage_details = " ".join([
            part.format(
                limit_amount=f"${random.randint(5000, 20000):,}",
                room_rent_percentage=random.randint(1, 2),
                ambulance_amount=f"${random.randint(500, 2000):,}",
                disease_list=random.choice(["cancer, heart diseases", "diabetes, kidney failure", "stroke, paralysis"])
            ) for part in coverage_details_parts
        ])

        records.append({
            "plan_id": i,
            "plan_name": plan_name,
            "provider_name": provider_name,
            "policy_terms": policy_terms,
            "coverage_details": coverage_details,
            "network_type": network_type
        })
    return records

def get_domain_specific_terms():
    """Returns domain-specific terms for fine-tuning context."""
    return DOMAIN_SPECIFIC_TERMS

if __name__ == "__main__":
    plans = generate_insurance_plans(num_plans=10)
    for plan in plans:
        print(plan)
    print("\nDomain Specific Terms:")
    for term, definition in get_domain_specific_terms().items():
        print(f"- {term}: {definition}")
