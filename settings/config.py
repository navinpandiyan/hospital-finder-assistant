from dotenv import load_dotenv
import spacy
from spacy.lang.en import English
import logging

# Load Environment Variables
load_dotenv()

# Load spaCy NER Model
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
    print("NER model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    NLP_MODEL = None
    
# Logging Configuration & Initialization
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("uvicorn.logger")

# Config Variables for Data Generation
CITY_COORDINATES = {
    # Abu Dhabi Emirate
    "Abu Dhabi": (24.4539, 54.3773),
    "Al Ain": (24.2075, 55.7447),
    "Madinat Zayed": (24.2480, 52.7070),
    "Ghayathi": (24.0000, 52.7833),
    "Liwa": (23.5000, 53.6500),
    "Ruways": (24.5700, 52.7800),
    "Sweihan": (24.1500, 55.7500),
    "Sila": (24.0100, 52.7400),
    "Habshan": (24.0500, 54.0000),
    "Marawah": (24.6000, 52.3000),

    # Dubai Emirate
    "Dubai": (25.2048, 55.2708),
    "Al Awir": (24.9500, 55.5000),
    "Lahbab": (24.9000, 55.6000),
    "Al Khawaneej": (25.2000, 55.4000),
    "Al Lisaili": (25.0500, 55.5500),
    "Al Rashidya": (25.2700, 55.3500),
    "Al Ruwayyah": (25.3200, 55.3200),

    # Sharjah Emirate
    "Sharjah": (25.3463, 55.4209),
    "Al Bataeh": (25.0000, 55.8000),
    "Al Hamriyah": (25.3500, 55.5500),
    "Al Jeer": (25.5000, 55.5800),
    "Mleiha": (25.1500, 55.7000),
    "Al Qor": (25.2000, 55.5000),
    "Al Hamraniyah": (25.4000, 55.5500),
    "Al Madam": (25.2000, 55.7000),

    # Ajman Emirate
    "Ajman": (25.4052, 55.5136),
    "Al Manama": (25.4400, 55.5100),
    "Masfut": (25.5500, 55.5200),
    "Falaj Al Mualla": (25.3700, 55.6100),

    # Fujairah Emirate
    "Fujairah": (25.1288, 56.3265),
    "Khor Fakkan": (25.3281, 56.3410),
    "Kalba": (25.1117, 56.3486),
    "Dibba Al Fujairah": (25.5691, 56.3542),
    "Masafi": (25.4394, 56.3411),
    "Al Badiyah": (25.3700, 56.3200),
    "Al Bithnah": (25.3500, 56.3300),
    "Al Qusaidat": (25.4200, 56.3700),
    "Huwaylat": (25.4300, 56.3600),
    "Mirbah": (25.4100, 56.3400),

    # Ras Al Khaimah Emirate
    "Ras Al Khaimah": (25.8007, 55.9762),
    "Digdaga": (25.7100, 55.9000),
    "Khatt": (25.7400, 56.0500),
    "Ghalilah": (25.6500, 56.0500),
    "Ghayl": (25.7200, 56.0000),
    "Khor Khwair": (25.6500, 55.9000),

    # Umm Al Quwain Emirate
    "Umm Al Quwain": (25.5660, 55.5530),

    # Other small settlements
    "Al Qor": (25.2100, 55.5100),
    "Al Jeer": (25.5000, 55.5800)
}

HOSPITAL_TYPES = [
    "allergy and immunology", "anesthesiology", "cardiology", "cardiothoracic surgery", "critical care", 
    "dermatology", "emergency medicine", "endocrinology", "family medicine", "gastroenterology", "general surgery", 
    "geriatrics", "hematology", "infectious disease", "internal medicine", "neonatology", "nephrology", "neurology", 
    "neurosurgery", "obstetrics", "gynecology", "oncology", "ophthalmology", "orthopedic", "otolaryngology", "ent", 
    "pediatrics", "physical medicine and rehabilitation", "plastic surgery", "psychiatry", "pulmonology", "radiology", 
    "rheumatology", "urology", "vascular surgery", "dental", "maternity", "cosmetic surgery", "sleep medicine", "pain management", 
    "sports medicine", "dialysis", "hepatology", "bariatric surgery", "palliative care", "neuropsychology", "infertility", "genetics", 
    "occupational medicine", "dermatologic surgery",
]

INSURANCE_PROVIDERS = [
    "aetna", "adnic", "daman", "metlife", "oman insurance", "nextcare", "takaful", "axa", "nas", 
    "orient", "bupa", "allianz", "mednet", "cigna", "etna", "universal health", "prudent", "gulf insurance", 
    "al safwa", "oman life", "etna life", "al manara", "medcare", "shield", "lifecare"
]


# Transcriber Config
TRANSCRIBER_OPENAI_MODEL = "whisper-1"
TRANSCRIBER_LANGUAGE = "en"


# Recognizer Config
FUZZY_MATCH_THRESHOLD = 95 # Fuzzy match threshold for SpaCy
USE_LLM_FOR_RECOGNITION = True
RECOGNIZER_MODEL = "google/gemini-2.0-flash-001"
RECOGNIZER_TEMPERATURE = 0.1

# Hospital Data Generation Config
HOSPITAL_DATA_FOLDER = "data"
HOSPITAL_DATA_FILE_NAME = "hospitals.sqlite"

# Clarifier Config
CLARIFIER_MODEL = "google/gemini-2.0-flash-001"
CLARIFIER_TEMPERATURE = 0.1

# Text-to-Dialogue Config
TEXT_TO_DIALOGUE = False
TEXT_TO_DIALOGUE_MODEL = "google/gemini-2.0-flash-001"
TEXT_TO_DIALOGUE_TEMPERATURE = 0.1

MAX_TURNS = 7

# Hospital Finder Config
N_HOSPITALS_TO_RETURN = 5
RATING_WEIGHT = 0.7