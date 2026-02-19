from typing import List

import spacy


class TratamentoSpacy:

    def __init__(self):
        self.__nlp = spacy.load("pt_core_news_sm")

    def __gerar_token(self, comentarios: List[str]) -> List[List[str]]:
        docs = self.__nlp.pipe(
            comentarios,
            batch_size=1000,
            n_process=1
        )

        return [
            [
                token.lemma_
                for token in doc
                if not token.is_stop
                   and not token.is_punct
                   and not token.like_num
                   and len(token.lemma_) > 2
            ]
            for doc in docs
        ]

    def executar_tratamento(self, comentarios: List[str]) -> List[List[str]]:
        return self.__gerar_token(comentarios)
