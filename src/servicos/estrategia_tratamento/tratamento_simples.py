import re
import emoji
from src.servicos.estrategia_tratamento.itratatamento import ITratamento


class TratamentoSimples(ITratamento[str, str]):

    @staticmethod
    def __remover_links(comentario: str) -> str:
        return re.sub(r"http\S+|www\S+|https\S+", "", comentario)

    @staticmethod
    def __remover_emoji(comentario: str) -> str:
        return emoji.replace_emoji(comentario, replace="")

    @staticmethod
    def __deixar_letras_minusculas(comentario: str) -> str:
        return comentario.lower()

    @staticmethod
    def __remover_mencoes(comentario: str) -> str:
        return re.sub(r"@\w+", "", comentario)

    @staticmethod
    def __remover_risadas(comentario: str) -> str:
        padrao = r'\b((k|h|a|u|e){2,}|(rs)+)\b'
        return re.sub(padrao, '', comentario, flags=re.IGNORECASE)

    @staticmethod
    def __remover_pontuacao(comentario: str) -> str:
        return re.sub(rf"[{re.escape(string.punctuation)}]", "", comentario)

    @staticmethod
    def __remover_acentos(comentario: str) -> str:
        comentario_normalizado = unicodedata.normalize('NFKD', comentario)
        comentario_ascii = comentario_normalizado.encode('ASCII', 'ignore').decode('utf-8')
        return comentario_ascii

    def executar_tratamento(self, comentario) -> str:
        comentario = str(comentario or "")
        comentario = self.__remover_links(comentario)
        comentario = self.__remover_emoji(comentario)
        comentario = self.__remover_mencoes(comentario)
        comentario = self.__deixar_letras_minusculas(comentario)
        comentario = self.__remover_risadas(comentario)
        comentario = self.__remover_pontuacao(comentario)
        comentario = self.__remover_acentos(comentario)
        comentario = re.sub(r"\s+", " ", comentario).strip()
        return comentario