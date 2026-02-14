from datetime import datetime
from typing import Optional

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.servico_s3.iservicos3 import Iservicos3


class GuardarDadosSteam3Corrente(Corrente):

    def __init__(self, servico_s3: Iservicos3):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__caminho_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__caminho_arquivo = f'steam/bronze/reviews_steam/'

    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        for dados in contexto.gerador_reviews_steam:

            caminho_completo = self.__caminho_arquivo + f'jogo_{dados["codigo_steam"]}/' + f'data_{self.__caminho_data}' + '_reviews.json'
            self.__servico_s3.guardar_dados(dados, caminho_completo)
        return True
