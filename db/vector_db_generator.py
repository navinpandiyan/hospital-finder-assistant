# db/vector_db_generator.py
# --------------------------
# Creates FAISS vector DB for hospitals

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from settings.config import HOSPITAL_DB_FOLDER, LOGGER, VECTOR_DB_FOLDER

# -----------------------------
# Prepare folder paths
# -----------------------------
os.makedirs(HOSPITAL_DB_FOLDER, exist_ok=True)
vector_db_path = os.path.abspath(os.path.join(HOSPITAL_DB_FOLDER, VECTOR_DB_FOLDER))

# -----------------------------
# Create FAISS vector DB
# -----------------------------
def create_vector_db_from_records(hospital_records, embedding_model=None):
    """
    Build and save FAISS vector DB from hospital records.
    Each record must include: hospital_name, address, location, hospital_type, insurance_providers, rating, latitude, longitude
    """
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    documents = []
    for rec in hospital_records:
        # Focus on content that aids semantic retrieval
        content = (
            f"{rec['hospital_name']} located in {rec['location']}. "
            f"Specialties: {', '.join(rec['hospital_type'])}. "
            f"Insurance accepted: {', '.join(rec['insurance_providers'])}. "
            f"Rating: {rec['rating']}."
        )

        metadata = {
            "hospital_id": rec["hospital_id"],
            "hospital_name": rec["hospital_name"],
            "address": rec["address"],
            "location": rec["location"],
            "latitude": rec["latitude"],
            "longitude": rec["longitude"],
            "hospital_type": rec["hospital_type"],
            "insurance_providers": rec["insurance_providers"],
            "rating": rec["rating"],
        }

        documents.append(Document(page_content=content, metadata=metadata))

    # Build FAISS DB
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(vector_db_path)

    LOGGER.info(f"✅ FAISS vector DB created at {vector_db_path} with {len(documents)} hospitals")
    return vector_db