from src.corrente_pipeline_comentarios.corrente import Corrente
from src.contexto.contexto import Contexto
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco


class GuardarDataFramePrata(Corrente):

    def __init__(self, operacoes_banco: IoperacoesBanco):
        super().__init__()
        self.__operacoes_banco = operacoes_banco

    def executar_processo(self, contexto: Contexto):
        dataframe_prata = contexto.dataframe_prata
        self.__operacoes_banco.guardar_dados(dataframe_prata)
        return True
