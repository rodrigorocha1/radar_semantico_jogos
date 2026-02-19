import re

import emoji

from src.servicos.estrategia_tratamento.itratatamento import ITratamento


class TratamentoSimples(ITratamento[str]):

    @staticmethod
    def __remover_links(comentario: str) -> str:
        comentario = re.sub(r"http\S+|www\S+|https\S+", "", comentario)
        return comentario

    @staticmethod
    def __remover_emoji(comentario: str) -> str:
        comentario = emoji.replace_emoji(comentario, replace="")
        return comentario

    @staticmethod
    def __deixar_letras_minusculas(comentario: str) -> str:
        comentario = comentario.lower()
        return comentario

    def executar_tratamento(self, comentario: str) -> str:
        comentario = self.__remover_links(comentario)
        comentario = self.__remover_emoji(comentario)
        comentario = self.__deixar_letras_minusculas(comentario)
        return comentario
