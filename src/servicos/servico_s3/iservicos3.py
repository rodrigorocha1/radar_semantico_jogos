from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class Iservicos3(ABC):

    @abstractmethod
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

    @abstractmethod
    def ler_jsons_para_dataframe(self, caminho_base: str) -> pd.DataFrame:
        pass