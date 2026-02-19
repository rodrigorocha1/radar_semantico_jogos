from typing import List, Tuple, Literal

import spacy

from src.servicos.estrategia_tratamento.itratatamento import ITratamento


class TratamentoSpacy(ITratamento):

    def __init__(self, comentario: str):
        self.__comentario = comentario
        self.__nlp = spacy.load("pt_core_news_sm")
        self.__docs = self.__nlp.pipe(self.__comentario, batch_size=1000, n_process=1)

    @property
    def comentario(self) -> str:
        return self.__comentario

    @comentario.setter
    def comentario(self, novo_comentario: str) -> None:
        self.__comentario = novo_comentario

    def __gerar_tokens(
            self
    ) -> Tuple[List[List[Tuple[str, Literal[False]]]], List[str]]:
        """
        Método para gerar tokens

        :return: tokens tratados
        :rtype: Tuple[List[List[Tuple[str, bool]]], List[str]]:
        """

        tokens_resultado = []
        comentarios_limpos = []

        for doc in self.__docs:
            tokens_filtrados = [
                (token.lemma_, token.is_punct)
                for token in doc
                if not token.is_stop
                   and not token.is_punct
                   and not token.like_num
                   and len(token.lemma_) > 2
            ]
            tokens_resultado.append(tokens_filtrados)

            # Gera comentário limpo unindo os lemmas filtrados
            comentarios_limpos.append(" ".join([lemma for lemma, _ in tokens_filtrados]))

        return tokens_resultado, comentarios_limpos

    def __gerar_entidades(self) -> List[List[Tuple[str, str]]]:
        """
        Gerar entidades nomeadas
        :param comentarios: comentários dos vídeos
        :type comentarios:  List[List[Tuple[str, str]]]
        :return: entidades nomeadas
        :rtype: str
        """

        entidades_resultado = []

        for doc in self.__docs:
            entidades = [(ent.text, ent.label_) for ent in doc.ents]
            entidades_resultado.append(entidades)

        return entidades_resultado

    def executar_tratamento(
            self
    ) -> Tuple[List[List[Tuple[str, Literal[False]]]], List[List[Tuple[str, str]]], List[str]]:
        """
        Executa os tratamentos dos comentários
        :param comentarios:comentários
        :type comentarios: str
        :return: comentários tratados
        :rtype: str
        """
        tokens, comentario_limpo = self.__gerar_tokens()
        entidades = self.__gerar_entidades()

        return tokens, entidades, comentario_limpo
