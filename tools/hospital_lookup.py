import asyncio
from typing import List, Optional
from pony.orm import db_session, select
from db.db import Hospital
import math
import pandas as pd

from settings.config import LOGGER

# -----------------------------
# Helper for distance calculation (Haversine formula)
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# -----------------------------
# Hospital Search Function
# -----------------------------
async def find_hospitals_async(
    user_lat: float,
    user_lon: float,
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    limit: int = 5,
    rating_weight: float = 0.7
) -> List[dict]:

    loop = asyncio.get_running_loop()

    @db_session
    def _query_hospitals():
        return list(select(h for h in Hospital))

    hospitals = await loop.run_in_executor(None, _query_hospitals)

    if not hospitals:
        return []

    # -----------------------------
    # Inner helper functions
    # -----------------------------
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def normalize(series):
        return (series - series.min()) / max(series.max() - series.min(), 1e-5)

    # -----------------------------
    # Convert DB objects to DataFrame
    # -----------------------------
    df = pd.DataFrame([{
        "hospital_id": h.hospital_id,
        "hospital_name": h.hospital_name,
        "location": h.location,
        "latitude": h.latitude,
        "longitude": h.longitude,
        "hospital_type": h.hospital_type.split(",") if isinstance(h.hospital_type, str) else h.hospital_type,
        "insurance_providers": h.insurance_providers.split(",") if isinstance(h.insurance_providers, str) else h.insurance_providers,
        "rating": float(getattr(h, "rating", 0))
    } for h in hospitals])

    # -----------------------------
    # Filter by types/insurance
    # -----------------------------
    if hospital_types:
        df = df[df["hospital_type"].apply(lambda types: any(ht.lower() in [t.lower() for t in types] for ht in hospital_types))]

    if insurance_providers and insurance_providers != ["mentioned"]:
        df = df[df["insurance_providers"].apply(lambda providers: any(ip.lower() in [p.lower() for p in providers] for ip in insurance_providers))]

    if df.empty:
        return []

    # -----------------------------
    # Compute distance, normalize, score, sort
    # -----------------------------
    df["distance_km"] = df.apply(lambda row: haversine_distance(user_lat, user_lon, row.latitude, row.longitude), axis=1)
    df["norm_rating"] = normalize(df["rating"])
    df["norm_distance"] = normalize(df["distance_km"])
    df["score"] = rating_weight * df["norm_rating"] + (1 - rating_weight) * (1 - df["norm_distance"])

    return df.sort_values("score", ascending=False).head(limit).to_dict("records")

# -----------------------------
# Hospital Lookup Wrapper
# -----------------------------
async def hospital_lookup_wrapper(
    user_lat: float,
    user_lon: float,
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    limit: int = 5,
    rating_weight: float = 0.7  # weight for rating vs distance in scoring
) -> List[dict]:
    """
    Wrapper around find_hospitals_async to lookup hospitals based on user's location,
    preferred hospital types, insurance providers, and limit. Returns a sorted list
    of hospitals scored by distance and rating.
    """
    LOGGER.info(
        f"Looking up hospitals for lat={user_lat}, lon={user_lon}, "
        f"types={hospital_types}, insurance={insurance_providers}"
    )

    hospitals = await find_hospitals_async(
        user_lat=user_lat,
        user_lon=user_lon,
        hospital_types=hospital_types,
        insurance_providers=insurance_providers,
        limit=limit,
        rating_weight=rating_weight
    )

    LOGGER.info(f"Found {len(hospitals)} hospitals matching criteria.")
    return hospitals
