# hospital_generator.py
# --------------------
# Generates synthetic hospital data for AI Voice Bot

import random
from settings.config import HOSPITAL_TYPES, INSURANCE_PROVIDERS, CITY_COORDINATES as CITY_COORDS

# Expanded hospital suffix options
HOSPITAL_SUFFIXES = [
    "Hospital", "Clinic", "Medical Center", "Center of Care", "Healthcare Center",
    "Medical Plaza", "Specialty Clinic", "Health Institute", "Wellness Center",
    "Emergency Center", "Care Hub", "Diagnostic Center", "Treatment Facility"
]

# Expanded city addresses for more variety
CITY_ADDRESSES = {
    # Abu Dhabi Emirate
    "Abu Dhabi": ["Corniche Rd", "Khalifa St", "Al Bateen St", "Electra St", "Al Maryah Island Rd", "Airport Rd", "Yas Island Blvd"],
    "Al Ain": ["Zayed Bin Sultan St", "Khalifa St", "Sheikh Hamed St", "Al Mutawaa St", "Al Jimi Rd"],
    "Madinat Zayed": ["Sheikh Khalifa St", "Al Dhahir St", "Al Raha St", "Zayed St"],
    "Ghayathi": ["Al Salamah St", "Al Bahar St", "Corniche Rd"],
    "Liwa": ["Liwa St", "Al Gharbia Rd", "Al Dhafra St"],
    "Ruways": ["Ruways St", "Coastal Rd", "Al Marfa St"],
    "Sweihan": ["Al Sweihan Rd", "Industrial St", "Main St"],
    "Sila": ["Sila St", "Al Shamkhah Rd", "Coastal Rd"],
    "Habshan": ["Habshan St", "Al Ain Rd", "Main Rd"],
    "Marawah": ["Marawah St", "Coastal Rd"],

    # Dubai Emirate
    "Dubai": ["Sheikh Zayed Rd", "Al Wasl Rd", "Jumeirah St", "Business Bay Blvd", "Downtown St", "Palm Jumeirah Ave", "Al Safa St"],
    "Al Awir": ["Al Awir Rd", "Industrial St", "Sheikh Mohammed Bin Zayed Rd"],
    "Lahbab": ["Lahbab Rd", "Desert Rd", "Camel St"],
    "Al Khawaneej": ["Al Khawaneej Rd", "Sheikh Mohammed Bin Zayed Rd", "Residential St"],
    "Al Lisaili": ["Al Lisaili St", "Palm St", "Industrial Rd"],
    "Al Rashidya": ["Al Rashidya Rd", "Airport Rd", "Residential St"],
    "Al Ruwayyah": ["Al Ruwayyah Rd", "Main St"],

    # Sharjah Emirate
    "Sharjah": ["Al Khan St", "King Faisal Rd", "University St", "Corniche St", "Al Majaz St", "Al Qasimia St"],
    "Al Bataeh": ["Al Bataeh Rd", "Main St", "Residential Rd"],
    "Al Hamriyah": ["Al Hamriyah St", "Corniche Rd", "Al Khan St"],
    "Al Jeer": ["Al Jeer Rd", "Port St", "Residential St"],
    "Mleiha": ["Mleiha Rd", "Desert Rd", "Al Falaj St"],
    "Al Qor": ["Al Qor Rd", "Residential St", "Corniche St"],
    "Al Hamraniyah": ["Al Hamraniyah St", "Al Jazeera St"],
    "Al Madam": ["Al Madam Rd", "Residential St", "Al Khaleej St"],

    # Ajman Emirate
    "Ajman": ["Al Ittihad St", "Corniche Rd", "Sheikh Rashid St", "Main St", "Al Jurf Rd"],
    "Al Manama": ["Al Manama St", "Corniche Rd", "Residential Rd"],
    "Masfut": ["Masfut Rd", "Al Ain Rd"],
    "Falaj Al Mualla": ["Falaj Al Mualla Rd", "Main St"],

    # Fujairah Emirate
    "Fujairah": ["Al Faseel Rd", "Corniche Rd", "Hamrah St", "Al Aqah Rd", "Al Dibba St"],
    "Khor Fakkan": ["Khor Fakkan Rd", "Corniche Rd", "Port St"],
    "Kalba": ["Kalba Rd", "Main St"],
    "Dibba Al Fujairah": ["Dibba Rd", "Coastal Rd"],
    "Masafi": ["Masafi Rd", "Industrial Rd"],
    "Al Badiyah": ["Al Badiyah Rd", "Main St"],
    "Al Bithnah": ["Al Bithnah Rd", "Residential St"],
    "Al Qusaidat": ["Al Qusaidat Rd", "Corniche Rd"],
    "Huwaylat": ["Huwaylat Rd", "Main St"],
    "Mirbah": ["Mirbah Rd", "Coastal Rd"],

    # Ras Al Khaimah Emirate
    "Ras Al Khaimah": ["Corniche Rd", "Khatt St", "Ghalilah Rd", "Main St"],
    "Digdaga": ["Digdaga Rd", "Industrial Rd", "Main St"],
    "Khatt": ["Khatt Rd", "Residential Rd"],
    "Ghalilah": ["Ghalilah Rd", "Port St"],
    "Ghayl": ["Ghayl Rd", "Residential St"],
    "Khor Khwair": ["Khor Khwair Rd", "Industrial St"],

    # Umm Al Quwain Emirate
    "Umm Al Quwain": ["Corniche Rd", "Al Sayegh Rd", "Main St"]
}


def generate_hospital_records(num_hospitals=150):
    """
    Generates a list of synthetic hospital records.
    Each hospital has:
        - hospital_id
        - hospital_name (dynamic using multiple specialties)
        - location (city)
        - latitude / longitude
        - hospital_type (1-3 specialties)
        - insurance_providers (1-3)
        - rating (numeric 1.0 - 5.0)
        - address (street + city)
    """
    CITIES = list(CITY_COORDS.keys())
    records = []
    
    for i in range(1, num_hospitals + 1):
        city = random.choice(CITIES)
        lat_base, lon_base = CITY_COORDS[city]
        latitude = lat_base + random.uniform(-0.05, 0.05)
        longitude = lon_base + random.uniform(-0.05, 0.05)

        # Hospital types & insurance
        specialties = random.sample(HOSPITAL_TYPES, k=random.randint(1, 3))
        insurances = random.sample(INSURANCE_PROVIDERS, k=random.randint(1, 3))

        # Name can include 1 or 2 specialties for more variety
        name_specialties = " & ".join(specialties[:random.randint(1, 2)]).title()
        hospital_name = f"{city} {name_specialties} " + random.choice(HOSPITAL_SUFFIXES)

        # Numeric rating
        rating = round(random.uniform(1.0, 5.0), 1)

        # Address: street number + street name
        streets = CITY_ADDRESSES.get(city, ["Main St", "Central Ave", "Sunset Blvd"])
        street = random.choice(streets)
        street_number = random.randint(1, 300)
        address = f"{street_number} {street}, {city}"

        records.append({
            "hospital_id": i,
            "hospital_name": hospital_name,
            "location": city,
            "latitude": latitude,
            "longitude": longitude,
            "hospital_type": specialties,
            "insurance_providers": insurances,
            "rating": rating,
            "address": address
        })
    
    return records
