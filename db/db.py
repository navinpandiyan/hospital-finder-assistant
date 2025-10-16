# db.py
# --------------------
# Sets up SQLite DB with Pony ORM, populates hospitals, and creates FAISS vector DB

import os
from pony.orm import Database, Required, db_session, count
from settings.config import HOSPITAL_DB_FILE_NAME, HOSPITAL_DB_FOLDER, VECTOR_DB_FOLDER, LOGGER
from db.hospital_generator import generate_hospital_records
from db.vector_db_generator import create_vector_db_from_records

# -----------------------------
# Create folders if not exists
# -----------------------------
os.makedirs(HOSPITAL_DB_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

db_folder = HOSPITAL_DB_FOLDER
os.makedirs(db_folder, exist_ok=True)
sqlite_db_path = os.path.abspath(os.path.join(db_folder, HOSPITAL_DB_FILE_NAME))
vector_db_path = os.path.abspath(os.path.join(db_folder, VECTOR_DB_FOLDER))

# -----------------------------
# Database Setup
# -----------------------------
db = Database()
db.bind(provider='sqlite', filename=sqlite_db_path, create_db=True)

# -----------------------------
# Hospital Model
# -----------------------------
class Hospital(db.Entity):
    hospital_id = Required(int, unique=True)
    hospital_name = Required(str)
    location = Required(str)
    latitude = Required(float)
    longitude = Required(float)
    address = Required(str)
    hospital_type = Required(str)
    insurance_providers = Required(str)
    rating = Required(float)

db.generate_mapping(create_tables=True)

# -----------------------------
# Populate DB if empty
# -----------------------------
with db_session:
    existing_count = count(h for h in Hospital)
    if existing_count == 0:
        LOGGER.info("Database empty. Generating synthetic hospital data...")
        # 1️⃣ Generate hospital records
        hospital_records = generate_hospital_records(num_hospitals=150)

        # 2️⃣ Populate SQLite DB
        for rec in hospital_records:
            Hospital(
                hospital_id=rec["hospital_id"],
                hospital_name=rec["hospital_name"],
                location=rec["location"],
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                address=rec["address"],
                hospital_type=",".join(rec["hospital_type"]),
                insurance_providers=",".join(rec["insurance_providers"]),
                rating=rec["rating"]
            )

        LOGGER.info(f"{len(hospital_records)} synthetic hospitals generated in {sqlite_db_path}")

    else:
        # If DB already has data, load existing records to possibly create vector DB
        hospital_records = [
            {
                "hospital_id": h.hospital_id,
                "hospital_name": h.hospital_name,
                "location": h.location,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "address": h.address,
                "hospital_type": h.hospital_type.split(","),
                "insurance_providers": h.insurance_providers.split(","),
                "rating": h.rating
            }
            for h in db.select("h from Hospital h")
        ]

# -----------------------------
# Create FAISS vector DB if missing
# -----------------------------
if not os.path.exists(vector_db_path):
    LOGGER.info("FAISS vector DB not found. Creating vector DB...")
    create_vector_db_from_records(hospital_records)
else:
    LOGGER.info(f"FAISS vector DB already exists at {vector_db_path}")
