# db.py
# --------------------
# Sets up SQLite DB with Pony ORM, populates hospitals, creates FAISS vector DB,
# and optionally fine-tunes the LLM on the hospital data.

import os
from pony.orm import Database, Required, db_session, count, select, Set
from settings.config import (
    HOSPITAL_DB_FILE_NAME,
    HOSPITAL_DB_FOLDER,
    VECTOR_DB_FOLDER,
    LOGGER,
    FINE_TUNE_OUTPUT_DIR,
    FINE_TUNE_DATA_PATH
)
from db.modules.hospital_generator import generate_hospital_records
from db.modules.insurance_generator import generate_insurance_plans
from db.modules.vector_db_generator import create_vector_db_from_records
from db.modules.fine_tune import fine_tune_insurance_llm
from db.modules.fine_tune_data_generator import generate_fine_tuning_data

# -----------------------------
# Create folders if not exists
# -----------------------------
os.makedirs(HOSPITAL_DB_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

sqlite_db_path = os.path.abspath(os.path.join(HOSPITAL_DB_FOLDER, HOSPITAL_DB_FILE_NAME))
vector_db_path = os.path.abspath(os.path.join(HOSPITAL_DB_FOLDER, VECTOR_DB_FOLDER))

# -----------------------------
# Database Setup
# -----------------------------
db = Database()
db.bind(provider='sqlite', filename=sqlite_db_path, create_db=True)

# -----------------------------
# Database Models
# -----------------------------
class Hospital(db.Entity):
    hospital_id = Required(int, unique=True)
    hospital_name = Required(str)
    location = Required(str)
    latitude = Required(float)
    longitude = Required(float)
    address = Required(str)
    hospital_type = Required(str)
    rating = Required(float)
    insurance_plans_link = Set('HospitalInsurancePlan') # New: Many-to-many relationship

class InsurancePlan(db.Entity):
    plan_id = Required(int, unique=True)
    plan_name = Required(str)
    provider_name = Required(str)
    policy_terms = Required(str) # Text field for detailed policy terms
    coverage_details = Required(str) # Text field for coverage specifics
    network_type = Required(str) # e.g., PPO, HMO, Exclusive
    rating = Required(float, default=0.0) # Added: rating field with default
    hospitals = Set('HospitalInsurancePlan') # New: Many-to-many relationship

class HospitalInsurancePlan(db.Entity):
    hospital = Required(Hospital)
    insurance_plan = Required(InsurancePlan)

db.generate_mapping(create_tables=True)

# -----------------------------
# Populate DB if empty
# -----------------------------
with db_session:
    existing_hospitals_count = count(h for h in Hospital)
    existing_insurance_plans_count = count(p for p in InsurancePlan)

    if existing_hospitals_count == 0:
        LOGGER.info("Hospital database empty. Generating synthetic hospital data...")
        # 1️⃣ Generate hospital records
        hospital_records = generate_hospital_records(num_hospitals=150)

        # 2️⃣ Populate SQLite DB with Hospitals
        for rec in hospital_records:
            Hospital(
                hospital_id=rec["hospital_id"],
                hospital_name=rec["hospital_name"],
                location=rec["location"],
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                address=rec["address"],
                hospital_type=",".join(rec["hospital_type"]),
                rating=rec["rating"]
            )
        LOGGER.info(f"{len(hospital_records)} synthetic hospitals generated in {sqlite_db_path}")
    else:
        LOGGER.info(f"{existing_hospitals_count} hospitals already exist in {sqlite_db_path}")


    if existing_insurance_plans_count == 0:
        LOGGER.info("Insurance Plan database empty. Generating synthetic insurance data...")
        # 1️⃣ Generate insurance plan records
        insurance_plans_data = generate_insurance_plans(num_plans=20)
        # 2️⃣ Populate SQLite DB with Insurance Plans
        for plan_data in insurance_plans_data:
            InsurancePlan(
                plan_id=plan_data["plan_id"],
                plan_name=plan_data["plan_name"],
                provider_name=plan_data["provider_name"],
                policy_terms=plan_data["policy_terms"],
                coverage_details=plan_data["coverage_details"],
                network_type=plan_data["network_type"],
                rating=plan_data["rating"] # Added rating to insurance plan data
            )
        LOGGER.info(f"{len(insurance_plans_data)} synthetic insurance plans generated.")

        # 3️⃣ Link hospitals to insurance plans
        hospitals = list(select(h for h in Hospital)) # Convert to list
        insurance_plans = list(select(ip for ip in InsurancePlan)) # Convert to list

        for hospital in hospitals:
            # Randomly assign a few insurance plans to each hospital
            import random
            num_plans_to_assign = random.randint(1, min(3, len(insurance_plans))) # Assign 1 to 3 plans
            assigned_plans = random.sample(insurance_plans, num_plans_to_assign)
            for plan in assigned_plans:
                HospitalInsurancePlan(hospital=hospital, insurance_plan=plan)
        LOGGER.info("Hospitals linked to insurance plans.")

    else:
        LOGGER.info(f"{existing_insurance_plans_count} insurance plans already exist.")

    # -----------------------------
    # Create FAISS vector DB if missing
    # -----------------------------
    if not os.path.exists(vector_db_path):
        # Load existing records from DB
        hospital_records = [
            {
                "hospital_id": h.hospital_id,
                "hospital_name": h.hospital_name,
                "location": h.location,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "address": h.address,
                "hospital_type": h.hospital_type.split(","),
                "insurance_providers": [link.insurance_plan.provider_name for link in h.insurance_plans_link], # Updated to get from new relation
                "rating": h.rating
            }
            for h in select(h for h in Hospital)
        ]
        LOGGER.info("FAISS vector DB not found. Creating vector DB...")
        create_vector_db_from_records(hospital_records)
    else: # This else corresponds to 'if not os.path.exists(vector_db_path):'
        LOGGER.info(f"FAISS vector DB already exists at {vector_db_path}")
        
    # -----------------------------
    # Generate fine-tuning data if not exists
    # -----------------------------
    if not os.path.exists(FINE_TUNE_DATA_PATH):
        hospital_records = [
            {
                "hospital_id": h.hospital_id,
                "hospital_name": h.hospital_name,
                "location": h.location,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "address": h.address,
                "hospital_type": h.hospital_type.split(","),
                "insurance_providers": [link.insurance_plan.provider_name for link in h.insurance_plans_link], # Updated to get from new relation
                "rating": h.rating
            }
            for h in select(h for h in Hospital)
        ]
        
        insurance_plans = [
            {
                "plan_id": i.plan_id,
                "plan_name": i.plan_name,
                "provider_name": i.provider_name,
                "policy_terms": i.policy_terms,
                "coverage_details": i.coverage_details,
                "network_type": i.network_type,
                "rating": i.rating
            }
            for i in select(i for i in InsurancePlan)
        ]
        
        LOGGER.info("Fine-tuning data not found. Generating fine-tuning data...")
        # Get actual Pony ORM entities for fine-tuning data generation
        all_hospitals = list(select(h for h in Hospital))
        all_insurance_plans = list(select(ip for ip in InsurancePlan))
        all_hospital_insurance_plans = list(select(hip for hip in HospitalInsurancePlan))

        LOGGER.info("Fine-tuning data not found. Generating fine-tuning data...")
        generate_fine_tuning_data(all_hospitals, all_insurance_plans, all_hospital_insurance_plans)
    else:
        LOGGER.info(f"Fine-tuning data already exists at {FINE_TUNE_DATA_PATH}")

    # -----------------------------
    # Fine-tune LLM on insurance data
    # -----------------------------
    if not os.path.exists(FINE_TUNE_OUTPUT_DIR):
        LOGGER.info("Fine-tuned LLM not found. Starting fine-tuning on insurance data...")
        fine_tune_insurance_llm(FINE_TUNE_DATA_PATH)
        LOGGER.info(f"Fine-tuned LLM saved to {FINE_TUNE_OUTPUT_DIR}")
    else:
        LOGGER.info(f"Fine-tuned LLM already exists at {FINE_TUNE_OUTPUT_DIR}")
