from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.estrategia_tratamento.processador_texto import ProcessadorTexto
from src.servicos.estrategia_tratamento.tratamento_simples import TratamentoSimples
from src.servicos.estrategia_tratamento.tratamento_spacy import TratamentoSpacy


class LimpezaComentariosCorrente(Corrente):
    def __init__(self):
        super().__init__()
        self.__processador_texto = ProcessadorTexto()

    def executar_processo(self, contexto: Contexto) -> bool:
        dataframe_original = contexto.dataframe_original
        self.__processador_texto.estrategia = TratamentoSimples()
        dataframe_original['comentario_limpo'] = dataframe_original['texto_comentario'].apply(
            self.__processador_texto.processar
        )
        self.__processador_texto.estrategia = TratamentoSpacy()
        dataframe_original['comentario_limpo'] = self.__processador_texto.processar(
            dataframe_original['comentario_limpo'].tolist()
        )
        print(dataframe_original.head())

        return True
