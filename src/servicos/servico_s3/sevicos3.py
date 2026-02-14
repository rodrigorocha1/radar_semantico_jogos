from typing import Dict

import boto3
from botocore.client import Config

from src.servicos.config.config import Config as c
from src.servicos.servico_s3.iservicos3 import Iservicos3


class ServicoS3(Iservicos3):

    def __init__(self):
        self.__boto3 = boto3.client(
            "s3",
            endpoint_url=c.MINIO_ENDPOINT,
            aws_access_key_id=c.MINIO_ACCESS_KEY,
            aws_secret_access_key=c.MINIO_SECRET_KEY,
            region_name=c.AWS_REGION,
            config=Config(signature_version="s3v4")
        )

    def guardar_dados(self, dados: Dict, caminho_arquivo: str):
        """
        Método para guardar dados

        :param dados: requisição da api
        :type dados: Dict
        :param caminho_arquivo: caminho do arquivo
        :type caminho_arquivo:  str
        :return:
        :rtype:
        """
        pass
