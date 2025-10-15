import asyncio
from typing import List, Optional
from pony.orm import db_session, select
from db.db import Hospital
from settings.config import LOGGER
import math
import pandas as pd

# -----------------------------
# Helper for distance calculation (Haversine formula)
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# -----------------------------
# Hospital Search Function
# -----------------------------
async def find_hospitals_async(
    user_lat: float,
    user_lon: float,
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    limit: int = 5
) -> pd.DataFrame:
    """
    Finds hospitals asynchronously using Pony ORM db_session,
    calculates distance, and returns a sorted pandas DataFrame.
    """
    # Pony ORM async support: run db_session in executor
    loop = asyncio.get_running_loop()

    @db_session
    def _query_hospitals():
        query = select(h for h in Hospital)

        # if hospital_types:
        #     query = query.filter(lambda h: any(ht.lower() in h.hospital_type.lower() for ht in hospital_types))

        # if insurance_providers:
        #     query = query.filter(lambda h: any(ip.lower() in h.insurance_providers.lower() for ip in insurance_providers))

        return list(query)

    hospitals = await loop.run_in_executor(None, _query_hospitals)

    # Convert to pandas DataFrame
    df = pd.DataFrame([{
        "hospital_id": h.hospital_id,
        "hospital_name": h.hospital_name,
        "location": h.location,
        "latitude": h.latitude,
        "longitude": h.longitude,
        "hospital_type": h.hospital_type,
        "insurance_providers": h.insurance_providers
    } for h in hospitals])
    
    # Filter by hospital types if provided
    df["hospital_type"] = df["hospital_type"].str.split(",")
    df["insurance_providers"] = df["insurance_providers"].str.split(",")
    
    select_df = df.copy()
    
    # Apply filters
    if hospital_types:
        select_df = select_df[select_df["hospital_type"].apply(lambda types: any(ht.lower() in [t.lower() for t in types] for ht in hospital_types))]
    
    if insurance_providers and insurance_providers != ["mentioned"]:
        select_df = select_df[select_df["insurance_providers"].apply(lambda providers: any(ip.lower() in [p.lower() for p in providers] for ip in insurance_providers))]
        
    # Calculate Haversine distance
    if not select_df.empty:
        select_df["distance_km"] = select_df.apply(
            lambda row: haversine_distance(user_lat, user_lon, row.latitude, row.longitude), axis=1
        )
        suggest_df = select_df.sort_values("distance_km").head(limit)
        return suggest_df.to_dict('records')
    return [] # Return an empty list if no hospitals are found

async def hospital_lookup_wrapper(
    user_lat: float,
    user_lon: float,
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    limit: int = 5
) -> List[dict]:
    """
    Looks up hospitals in the database based on user's location, desired hospital types,
    and insurance providers. Returns a list of matching hospitals sorted by distance.
    """
    LOGGER.info(f"Looking up hospitals for lat={user_lat}, lon={user_lon}, types={hospital_types}, insurance={insurance_providers}")
    
    # The find_hospitals_in_db function uses Pony ORM's db_session, which is synchronous.
    # We run it in a separate thread to avoid blocking the asyncio event loop.
    hospitals = await find_hospitals_async(
        user_lat=user_lat,
        user_lon=user_lon,
        hospital_types=hospital_types,
        insurance_providers=insurance_providers,
        limit=limit
    )
    
    LOGGER.info(f"Found {len(hospitals)} hospitals.")
    return hospitals

# ------------------------------
# Example usage
# ------------------------------
# async def main():
#     user_lat = 25.443709994636056
#     user_lon = 55.509422409908356
#     df_hospitals = await find_hospitals_async(user_lat, user_lon, hospital_types=[], insurance_providers=["nas"], limit=5)
#     print(df_hospitals)


# if __name__ == "__main__":
#     asyncio.run(main())
