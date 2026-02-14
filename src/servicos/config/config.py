import os
from typing import Final, Dict

from dotenv import load_dotenv

load_dotenv()


class Config:
    STEAM_API_URL: Final[str] = os.getenv("STEAM_API_URL", " ")
    MINIO_ENDPOINT: Final[str] = os.getenv("MINIO_HOST_URL", " ")
    MINIO_ACCESS_KEY: Final[str] = os.getenv("MINIO_ROOT_USER", " ")
    MINIO_SECRET_KEY: Final[str] = os.getenv("MINIO_ROOT_PASSWORD", " ")
    MINIO_BUCKET: Final[str] = os.getenv("MINIO_BUCKET", "")
    AWS_REGION: Final[str] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    MINIO_BUCKET_PLN: Final[str] = os.getenv("MINIO_BUCKET_PLN", "")
    MINIO_HOST_URL_DUCKDB: Final[str] = os.getenv("MINIO_HOST_URL_DUCKDB", "")
    CHAVE_API_YOUTUBE: Final[str] = os.getenv("CHAVE_API_YOUTUBE", "")
    CONFIG_JOGOS: Final[Dict[str, int]] = {
        "star_rupture": 1631270,
        "no_mans_sky": 275850,
        "x4_foundations": 392160,
        "satisfactory": 526870,
        "planet_crafter": 1284190,
        "the_planet_crafter_toxicity": 4078590,
        "the_planet_crafter_planet_humble": 3142540,
        "elite_dangerous": 359320,
        "elite_dangerous_odissey": 1336350,
        "euro_truck_simulator": 227300,
        "euro_truck_simulator_grecia": 2604420,
        "euro_truck_simulator_iberia": 1209460,
        "euro_truck_simulator_italia": 558244,
        "euro_truck_simulator_beyound_the_baltic_sea": 925580,
        "euro_truck_simulator_road_to_the_black_sea": 1056760,
        "euro_truck_simulator_vive_le_france": 531130,
        "euro_truck_simulator_scandinaavia": 304212,
        "euro_truck_simulator_going_east": 227310,
        "euro_truck_simulator_nordic_horizons": 2780810,
        "space_engineers": 244850,
        "cities_skylines": 255710,
        "cities_skylines_dois": 949230,
    }
