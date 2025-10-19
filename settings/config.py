from dotenv import load_dotenv
import spacy
import logging

# ----------------------------
# App Mode
# ----------------------------
MODE = "voicebot"  # or "voicebot"

# Load environment variables
load_dotenv()

# ----------------------------
# Logging Configuration
# ----------------------------
if MODE == "chatbot" or MODE == "voicebot":
    # Disable all logging output
    logging.disable(logging.CRITICAL)
    LOGGER = logging.getLogger("app.log")
else:
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger("app.log")

# Load spaCy NER Model
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
    LOGGER.info("NER model 'en_core_web_sm' loaded successfully.")
except OSError:
    LOGGER.error("Error: spaCy model 'en_core_web_sm' not found.")
    LOGGER.warning("Please run: python -m spacy download en_core_web_sm")
    NLP_MODEL = None

# Silence httpx and httpcore info/debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
HOSPITAL_DB_FOLDER = "data"
HOSPITAL_DB_FILE_NAME = "hospitals.sqlite"
VECTOR_DB_FOLDER = "vdb_hospitals"


# Clarifier Config
CLARIFIER_MODEL = "google/gemini-2.0-flash-001"
CLARIFIER_TEMPERATURE = 0.1

# Text-to-Dialogue Config
TEXT_TO_DIALOGUE = False
TEXT_TO_DIALOGUE_MODEL = "google/gemini-2.0-flash-001"
TEXT_TO_DIALOGUE_TEMPERATURE = 0.1

MAX_TURNS = 7

# Hospital Finder Config
DEFAULT_N_HOSPITALS_TO_RETURN = 5
DEFAULT_DISTANCE_KM = 30000

LOOKUP_MODE = "rag" # "simple" / "rag"
RAG_GROUNDER_MODEL = "google/gemini-2.0-flash-001"
RAG_GROUNDER_TEMPERATURE = 0.1
GROUND_WITH_FINE_TUNE = True


# -----------------------------
# -----------------------------
# LLM Fine-Tuning Configurations
# -----------------------------
# -----------------------------

# -----------------------------
# Base Model Configurations
# -----------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct"   # Hugging Face Mistral instruct model
TOKENIZER_MODEL = BASE_MODEL                    # Usually same as base model

# -----------------------------
# Output / Data Paths
# -----------------------------
FINE_TUNE_OUTPUT_DIR = "data/rag_llm"          # Where fine-tuned model will be saved
FINE_TUNE_DATA_PATH = "db/insurance_data.json" # Fine-tuning data

# -----------------------------
# Training Hyperparameters
# -----------------------------
BATCH_SIZE = 1                                 # Per-device batch size (lower for 7B models)
EPOCHS = 3                                     # Fine-tuning epochs
LEARNING_RATE = 2e-4                           # Learning rate
MAX_SEQ_LEN = 512                               # Max token length for inputs

# Gradient accumulation for effective batch size
GRADIENT_ACCUMULATION_STEPS = 8               # Increase for memory-constrained GPUs

# Mixed precision
FP16 = True

# -----------------------------
# Logging / Checkpoints
# -----------------------------
SAVE_STEPS = 50                                # Save checkpoint every N steps
LOGGING_STEPS = 20                              # Log every N steps

# -----------------------------
# QLoRA / LoRA Configuration
# -----------------------------
LOAD_IN_4BIT = True                            # 4-bit quantization
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]         # Mistral uses 'q_proj' and 'v_proj' in transformer layers
LORA_TASK_TYPE = "CAUSAL_LM"

# -----------------------------
# Optional / Advanced
# -----------------------------
USE_SAFETENSORS = True                         # Save model in safe serialization
OPTIMIZER = "adamw_torch"
GRADIENT_CHECKPOINTING = True                  # Save memory during training