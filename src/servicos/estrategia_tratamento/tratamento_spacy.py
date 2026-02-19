from typing import List

import spacy


class TratamentoSpacy:

    def __init__(self):
        self.__nlp = spacy.load("pt_core_news_sm")


    def __gerar_token(self, comentario: str) -> List[str]:
        docs = self.__nlp.pipe(
            comentario,
            batch_size=1000,
            n_process=1
        )
        tokens_filtrados = filter(
            lambda token: (
                    not token.is_stop
                    and not token.is_punct
                    and not token.like_num
                    and len(token.lemma_) > 2
            ),
            docs,
        )

        return list(map(lambda token: token.lemma_, tokens_filtrados))

    def executar_tratamento(self, comentario: str) -> List[str]:
        tokens = self.__gerar_token(comentario)
        return tokens
