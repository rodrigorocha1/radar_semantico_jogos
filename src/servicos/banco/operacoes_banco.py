import os
from datetime import datetime

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
        self.__caminho_s3_prata = f's3://{Config.MINIO_BUCKET_PLN}/comentarios/prata/comentarios_limpos_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> pd.DataFrame:
        extensao = os.path.splitext(caminho_consulta)[1].lower()
        print(f"Extensão do arquivo: {extensao}")

        if extensao == ".json":
            reader = "read_json_auto(?)"
        elif extensao == ".csv":
            reader = "read_csv_auto(?)"
        else:
            raise ValueError(f"Formato não suportado: {extensao}")

        query = f"""
            SELECT *
            FROM {reader}
            WHERE {id_consulta}
        """

        result = self.__con.execute(query, [caminho_consulta])
        df = result.fetchdf()
        return df

    def guardar_dados(self, dados: pd.DataFrame):
        self.__con.register('df_temp', dados)
        self.__con.execute(
            f'COPY df_temp TO "{self.__caminho_s3_prata}" (FORMAT CSV, HEADER TRUE,  DELIMITER "|")')


if __name__ == '__main__':
    from src.servicos.config.config import Config as c

    obdb = OperacoesBancoDuckDb()
    df = obdb.consultar_dados(
        "1=1",
        caminho_consulta=f's3://{c.MINIO_BUCKET_PLN}/youtube/bronze/resposta_comentarios_youtube/**/*.json'
    )
    print(df)
