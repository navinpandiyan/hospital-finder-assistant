# hospital_generator.py
# --------------------
# Generates synthetic hospital data for AI Voice Bot
# Can be imported and used in db.py

import random
from settings.config import HOSPITAL_TYPES, INSURANCE_PROVIDERS, CITY_COORDINATES as CITY_COORDS

def generate_hospital_records(num_hospitals=150):
    """
    Generates a list of synthetic hospital records.
    Each hospital has:
        - hospital_id
        - hospital_name
        - location (city)
        - latitude / longitude
        - hospital_type (1-3 specialties)
        - insurance_providers (1-3)
    """
    CITIES = list(CITY_COORDS.keys())
    suffixes = ["Hospital", "Center", "Center of Care", "Medical Center", "Clinic"]
    records = []
    for i in range(1, num_hospitals + 1):
        city = random.choice(CITIES)
        lat_base, lon_base = CITY_COORDS[city]
        latitude = lat_base + random.uniform(-0.05, 0.05)
        longitude = lon_base + random.uniform(-0.05, 0.05)
        specialties = random.sample(HOSPITAL_TYPES, k=random.randint(1, 3))
        insurances = random.sample(INSURANCE_PROVIDERS, k=random.randint(1, 3))
        hospital_name = f"{city} {specialties[0].capitalize()} ".title() + random.choice(suffixes)

        records.append({
            "hospital_id": i,
            "hospital_name": hospital_name,
            "location": city,
            "latitude": latitude,
            "longitude": longitude,
            "hospital_type": specialties,
            "insurance_providers": insurances
        })
    return records