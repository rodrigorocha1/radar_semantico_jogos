import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.estrategia_tratamento.processador_texto import ProcessadorTexto
from src.servicos.estrategia_tratamento.tratamento_simples import TratamentoSimples
from src.servicos.estrategia_tratamento.tratamento_spacy import TratamentoSpacy

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class LimpezaComentariosCorrente(Corrente):
    def __init__(self):
        super().__init__()
        self.__processador_texto = ProcessadorTexto()

    def executar_processo(self, contexto: Contexto) -> bool:
        dataframe_original = contexto.dataframe_original
        self.__processador_texto.estrategia = TratamentoSimples(comentario="")
        dataframe_original['comentario_limpo'] = self.__processador_texto.processar(
            dados=dataframe_original['texto_comentario'].tolist()
        )
        print(dataframe_original['comentario_limpo'].tolist())


        self.__processador_texto.estrategia = TratamentoSpacy(comentario="")
        a = self.__processador_texto.processar(
            dataframe_original['comentario_limpo']
        )

        print(a[0])

        # # Cria colunas dinâmicas
        # dataframe_original['lemma'] = [[t[0] for t in token_list] for token_list in tokens_resultado]
        # dataframe_original['punct'] = [[t[1] for t in token_list] for token_list in tokens_resultado]
        # dataframe_original['entidades_texto'] = [[e[0] for e in ent_list] for ent_list in entidades_resultado]
        # dataframe_original['entidades_label'] = [[e[1] for e in ent_list] for ent_list in entidades_resultado]
        # dataframe_original['comentarios_limpos'] =
        # print(dataframe_original.loc[6])

        return True
