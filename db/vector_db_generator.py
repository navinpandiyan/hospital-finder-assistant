import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from settings.config import HOSPITAL_DB_FOLDER, LOGGER, VECTOR_DB_FOLDER

# -----------------------------
# Create folder if not exists
# -----------------------------
db_folder = HOSPITAL_DB_FOLDER
os.makedirs(db_folder, exist_ok=True)
vector_db_path = os.path.abspath(os.path.join(db_folder, VECTOR_DB_FOLDER))

def create_vector_db_from_records(hospital_records, embedding_model=None):
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()  # <-- no raw client

    documents = []
    for rec in hospital_records:
        content = f"{rec['hospital_name']} | {rec['address']} | {rec['location']} | " \
                  f"Specialties: {','.join(rec['hospital_type'])} | " \
                  f"Insurance: {','.join(rec['insurance_providers'])} | " \
                  f"Rating: {rec['rating']} | Latitude: {rec['latitude']} | Longitude: {rec['longitude']}"

        metadata = {
            "hospital_id": rec["hospital_id"],
            "hospital_name": rec["hospital_name"],
            "address": rec["address"],
            "city": rec["location"],
            "latitude": rec["latitude"],
            "longitude": rec["longitude"],
            "hospital_type": rec["hospital_type"],
            "insurance_providers": rec["insurance_providers"],
            "rating": rec["rating"]
        }

        documents.append(Document(page_content=content, metadata=metadata))

    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(vector_db_path)

    LOGGER.info(f"FAISS vector DB created at {vector_db_path} with {len(documents)} hospitals")
    return vector_db
