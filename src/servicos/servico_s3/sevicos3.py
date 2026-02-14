import json
from typing import Dict, List

import boto3
import pandas as pd
import s3fs
from botocore.client import Config

from src.servicos.config.config import Config as c
from src.servicos.servico_s3.iservicos3 import Iservicos3

pd.set_option("display.max_columns", None)

# Não quebrar linha no meio
pd.set_option("display.expand_frame_repr", False)

# Aumentar largura do display
pd.set_option("display.width", 200)

# Mostrar texto completo (ou limite maior)
pd.set_option("display.max_colwidth", 300)

# Mostrar mais linhas
pd.set_option("display.max_rows", 20)


class ServicoS3(Iservicos3):

    def __init__(self):
        self.__cliente_s3 = self._criar_cliente()
        self.__fs = self.criar_filesystem()

    @staticmethod
    def criar_filesystem():
        return s3fs.S3FileSystem(
            key=c.MINIO_ACCESS_KEY,
            secret=c.MINIO_SECRET_KEY,
            client_kwargs={"endpoint_url": c.MINIO_ENDPOINT},
        )

    def guardar_dados(self, dados: Dict, caminho_arquivo: str) -> None:

        linhas = self._obter_linhas_existentes(caminho_arquivo)

        nova_linha = self._serializar_dados(dados)
        linhas.append(nova_linha)

        self._salvar_linhas(caminho_arquivo, linhas)

    @staticmethod
    def _criar_cliente():
        return boto3.client(
            "s3",
            endpoint_url=c.MINIO_ENDPOINT,
            aws_access_key_id=c.MINIO_ACCESS_KEY,
            aws_secret_access_key=c.MINIO_SECRET_KEY,
            region_name=c.AWS_REGION,
            config=Config(signature_version="s3v4")
        )

    def _obter_linhas_existentes(self, caminho_arquivo: str) -> List[str]:
        try:
            obj = self.__cliente_s3.get_object(
                Bucket=c.MINIO_BUCKET_PLN,
                Key=caminho_arquivo
            )

            conteudo = obj['Body'].read().decode('utf-8')
            return [
                linha for linha in conteudo.splitlines()
                if linha.strip()
            ]

        except self.__cliente_s3.exceptions.NoSuchKey:
            return []

    @staticmethod
    def _serializar_dados(dados: Dict) -> str:
        return json.dumps(dados, ensure_ascii=False)

    def _salvar_linhas(self, caminho_arquivo: str, linhas: List[str]) -> None:
        novo_conteudo = "\n".join(linhas)

        self.__cliente_s3.put_object(
            Bucket=c.MINIO_BUCKET_PLN,
            Key=caminho_arquivo,
            Body=novo_conteudo.encode('utf-8'),
            ContentType="application/json"
        )
