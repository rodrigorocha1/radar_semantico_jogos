from abc import ABC, abstractmethod

import pandas as pd


class IoperacoesBanco(ABC):

    @abstractmethod
    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> pd.DataFrame:
        """
        Método para consultar registro inseridos
        :param id_consulta: id da consulta, chave
        :type id_consulta: str
        :param caminho_consulta: caminho do minio s3
        :type caminho_consulta: str
        :return: dataframe com os resultados
        :rtype: pd.Dataframe
        """
        pass