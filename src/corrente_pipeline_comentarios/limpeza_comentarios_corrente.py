from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.estrategia_tratamento.processador_texto import ProcessadorTexto
from src.servicos.estrategia_tratamento.tratamento_simples import TratamentoSimples
from src.servicos.estrategia_tratamento.tratamento_spacy import TratamentoSpacy
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


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
        tokens_resultado, entidades_resultado, comentario_limpo, embedings = self.__processador_texto.processar(
            dataframe_original['comentario_limpo'].tolist()
        )


        dataframe_original['lemma'] = [[t[0] for t in token_list] for token_list in tokens_resultado]
        dataframe_original['punct'] = [[t[1] for t in token_list] for token_list in tokens_resultado]
        dataframe_original['entidades_texto'] = [[e[0] for e in ent_list] for ent_list in entidades_resultado]
        dataframe_original['entidades_label'] = [[e[1] for e in ent_list] for ent_list in entidades_resultado]
        dataframe_original['comentarios_limpos'] = comentario_limpo
        dataframe_original['embedings'] = embedings

        contexto.dataframe_prata = dataframe_original
        return True
