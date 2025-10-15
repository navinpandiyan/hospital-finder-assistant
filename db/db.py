# db.py
# --------------------
# Sets up SQLite DB with Pony ORM and populates hospitals

import os
from pony.orm import Database, Required, db_session, count
from settings.config import HOSPITAL_DATA_FOLDER, HOSPITAL_DATA_FILE_NAME, LOGGER
from db.hospital_generator import generate_hospital_records

# -----------------------------
# Create folder if not exists
# -----------------------------
db_folder = HOSPITAL_DATA_FOLDER
os.makedirs(db_folder, exist_ok=True)
db_file = os.path.abspath(os.path.join(db_folder, HOSPITAL_DATA_FILE_NAME))

# -----------------------------
# Database Setup
# -----------------------------
db = Database()
db.bind(provider='sqlite', filename=db_file, create_db=True)

# -----------------------------
# Hospital Model
# -----------------------------
class Hospital(db.Entity):
    hospital_id = Required(int, unique=True)
    hospital_name = Required(str)
    location = Required(str)
    latitude = Required(float)
    longitude = Required(float)
    hospital_type = Required(str)        # comma-separated specialties
    insurance_providers = Required(str)  # comma-separated providers

db.generate_mapping(create_tables=True)

# -----------------------------
# Populate DB if empty
# -----------------------------
with db_session:
    existing_count = count(h for h in Hospital)
    if existing_count == 0:
        LOGGER.info("Database empty. Generating synthetic hospital data...")
        hospital_records = generate_hospital_records(num_hospitals=150)

        for rec in hospital_records:
            hosp = Hospital(
                hospital_id=rec["hospital_id"],
                hospital_name=rec["hospital_name"],
                location=rec["location"],
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                hospital_type=",".join(rec["hospital_type"]),
                insurance_providers=",".join(rec["insurance_providers"])
            )

        LOGGER.info(f"{len(hospital_records)} synthetic hospitals generated in {db_file}")
