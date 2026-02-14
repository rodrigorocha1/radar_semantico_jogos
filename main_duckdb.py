import duckdb
from src.servicos.config.config import Config
import pandas as pd

# Mostra todas as colunas
pd.set_option('display.max_columns', None)

# Ajusta largura do terminal para caber as colunas
pd.set_option('display.width', 3000)

# Número de linhas a exibir
pd.set_option('display.max_rows', 10)

# Formatação de floats
pd.set_option('display.float_format', '{:.2f}'.format)


def read_from_minio(path: str, file_type: str = "json", steamid_to_filter: str = None):
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(f"""
        SET s3_region='{Config.AWS_REGION}';
        SET s3_access_key_id='{Config.MINIO_ACCESS_KEY}';
        SET s3_secret_access_key='{Config.MINIO_SECRET_KEY}';
        SET s3_endpoint='{Config.MINIO_HOST_URL_DUCKDB}';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
    """)

    if file_type == "json":
        if steamid_to_filter:
            query = f"""
                SELECT *
                FROM read_json_auto('{path}')
                WHERE author.steamid = '{steamid_to_filter}'
            """
        else:
            query = f"SELECT * FROM read_json_auto('{path}')"
    elif file_type == "csv":
        query = f"SELECT * FROM read_csv_auto('{path}')"
    elif file_type == "parquet":
        query = f"SELECT * FROM read_parquet('{path}')"
    else:
        raise ValueError("file_type deve ser 'json', 'csv' ou 'parquet'")

    df = con.execute(query).fetchdf()
    return df
file_path = "s3://extracao/steam/bronze/reviews_steam/jogo_1631270/*.json"
steamid_to_filter = "76561197979398614"

df = read_from_minio(file_path, file_type="json", steamid_to_filter=steamid_to_filter)
print(df.head())