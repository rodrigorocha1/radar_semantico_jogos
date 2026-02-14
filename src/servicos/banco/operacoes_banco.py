import duckdb
import pandas as pd

from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.config import Config


class OperacoesBancoDuckDb(IoperacoesBanco):

    def __init__(self):
        self.__con = duckdb.connect()
        self.__con.execute("INSTALL httpfs")
        self.__con.execute("LOAD httpfs")
        self.__con.execute(f"""
                SET s3_region='{Config.AWS_REGION}';
                SET s3_access_key_id='{Config.MINIO_ACCESS_KEY}';
                SET s3_secret_access_key='{Config.MINIO_SECRET_KEY}';
                SET s3_endpoint='{Config.MINIO_HOST_URL_DUCKDB}';
                SET s3_use_ssl=false;
                SET s3_url_style='path';
            """)

    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM read_json_auto(?)
            WHERE {id_consulta}
        """

        # Apenas caminho_consulta como parâmetro
        result = self.__con.execute(query, [caminho_consulta])
        df = result.fetchdf()
        return df