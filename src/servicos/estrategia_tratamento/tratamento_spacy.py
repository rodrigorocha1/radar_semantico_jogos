from typing import List, Tuple
import spacy

class TratamentoSpacy:

    def __init__(self):
        # Carrega modelo de português
        self.__nlp = spacy.load("pt_core_news_lg")

    def __gerar_tokens(
        self, comentarios: List[str]
    ) -> Tuple[List[List[Tuple[str, bool]]], List[str]]:
        """
        Método para gerar tokens
        :param comentarios: comentários de vídeos steam
        :type comentarios:  str
        :return: tokens tratados
        :rtype: Tuple[List[List[Tuple[str, bool]]], List[str]]
        """
        docs = self.__nlp.pipe(comentarios, batch_size=1000, n_process=1)
        tokens_resultado = []
        comentarios_limpos = []

        for doc in docs:
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

    def __gerar_entidades(self, comentarios: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Gerar entidades nomeadas
        :param comentarios: comentários dos vídeos
        :type comentarios:  List[List[Tuple[str, str]]]
        :return: entidades nomeadas
        :rtype: str
        """
        docs = self.__nlp.pipe(comentarios, batch_size=1000, n_process=1)
        entidades_resultado = []

        for doc in docs:
            entidades = [(ent.text, ent.label_) for ent in doc.ents]
            entidades_resultado.append(entidades)

        return entidades_resultado


    def __gerar_embedding(self, comentarios: List[str]) -> List[List[float]]:
        docs = self.__nlp.pipe(comentarios, batch_size=1000, n_process=1)
        embeddings = [doc.vector.tolist() for doc in docs]
        return embeddings

    def executar_tratamento(
        self, comentarios: List[str]
    ) -> Tuple[List[List[Tuple[str, bool]]], List[List[Tuple[str, str]]], List[str], List[List[float]]]:
        """
        Executa os tratamentos dos comentários
        :param comentarios:comentários
        :type comentarios: str
        :return: comentários tratados
        :rtype:  Tuple[List[List[Tuple[str, bool]]], List[List[Tuple[str, str]]], List[str], List[List[float]]
        """
        tokens, comentario_limpo = self.__gerar_tokens(comentarios)
        entidades = self.__gerar_entidades(comentarios)
        embeddings = self.__gerar_embedding(comentarios)

        return tokens, entidades, comentario_limpo, embeddings
