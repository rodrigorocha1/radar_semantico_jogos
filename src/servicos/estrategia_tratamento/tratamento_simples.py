import re
import emoji

from src.servicos.estrategia_tratamento.itratatamento import ITratamento


class TratamentoSimples(ITratamento[str]):

    def __init__(self, comentario: str):
        self.__comentario = comentario

    @property
    def comentario(self) -> str:
        return self.__comentario

    @comentario.setter
    def comentario(self, novo_comentario: str) -> None:
        self.__comentario = novo_comentario

    def __remover_links(self) -> str:
        return re.sub(r"http\S+|www\S+|https\S+", "", self.__comentario)

    def __remover_emoji(self) -> str:
        return emoji.replace_emoji(self.__comentario, replace="")

    def __deixar_letras_minusculas(self) -> str:
        return self.__comentario.lower()

    def __remover_mencoes(self) -> str:
        return re.sub(r"@\w+", "", self.__comentario)

    def executar_tratamento(self) -> str:
        self.__comentario = self.__remover_links()
        self.__comentario = self.__remover_emoji()
        self.__comentario = self.__remover_mencoes()
        self.__comentario = self.__deixar_letras_minusculas()
        return self.__comentario