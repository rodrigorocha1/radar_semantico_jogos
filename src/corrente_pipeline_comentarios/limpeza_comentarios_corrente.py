from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.estrategia_tratamento.processador_texto import ProcessadorTexto


class LimpezaComentariosCorrente(Corrente):
    def __init__(self):
        super().__init__()
        self.__processador_texto = ProcessadorTexto()

    def executar_processo(self, contexto: Contexto) -> bool:
        dataframe_original = contexto.dataframe_original


        return True
