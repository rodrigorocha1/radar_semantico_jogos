from typing import Dict

import boto3
from botocore.client import Config

from src.servicos.config.config import Config as c
from src.servicos.servico_s3.iservicos3 import Iservicos3


class ServicoS3(Iservicos3):

    def __init__(self):
        self.__cliente_s3 = boto3.client(
            "s3",
            endpoint_url=c.MINIO_ENDPOINT,
            aws_access_key_id=c.MINIO_ACCESS_KEY,
            aws_secret_access_key=c.MINIO_SECRET_KEY,
            region_name=c.AWS_REGION,
            config=Config(signature_version="s3v4")
        )

    from typing import Dict

    def guardar_dados(self, dados: Dict, caminho_arquivo: str):
        """
        Método para guardar dados em formato JSON, adicionando no final do arquivo.

        :param dados: requisição da api
        :type dados: Dict
        :param caminho_arquivo: caminho do arquivo no bucket
        :type caminho_arquivo: str
        """
        try:

            obj = self.__cliente_s3.get_object(Bucket=c.MINIO_BUCKET_PLN, Key=caminho_arquivo)
            conteudo_existente = obj['Body'].read().decode('utf-8')
            linhas = [linha for linha in conteudo_existente.splitlines() if linha.strip()]
        except self.__cliente_s3.exceptions.NoSuchKey:

            linhas = []

        novo_conteudo = "\n".join(linhas)

        self.__cliente_s3.put_object(
            Bucket=c.MINIO_BUCKET_PLN,
            Key=caminho_arquivo,
            Body=novo_conteudo.encode('utf-8'),
            ContentType="application/json"
        )


if __name__ == '__main__':
    meu_json = {
        "nome": "Rodrigo",
        "idade": 30,
        "profissao": "desenvolvedor"
    }

    # Caminho do arquivo no MinIO
    arquivo_s3 = "dados/usuario.json"

    ss3 = ServicoS3()
    ss3.guardar_dados(meu_json, arquivo_s3)
