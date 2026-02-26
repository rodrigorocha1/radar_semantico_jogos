from typing import Dict, List, Literal, Optional, Tuple

import databricks
import tensorflow as tf


class SOMV2(tf.Module):

    def __init__(
        self,
        linhas: int,
        colunas: int,
        dimensao: int,
        taxa_aprendizado: float = 0.5,
        sigma: Optional[float] = None,
        metrica: Literal["euclidiana", "cosseno"] = "euclidiana",
        inicializacao: Literal["random", "pca"] = "random",
        log_dir: Optional[str] = None

    ):
        self.__linhas = linhas
        self.__colunas = colunas
        self.__dimensao = dimensao
        self.__sigma = sigma
        self.__metrica = metrica
        self.__inicializacao: Literal["random",
                                      "sample", "pca"] = inicializacao
        self.__log_dir = log_dir

    def iniciar_pesos(self, dados: Optional[tf.Tensor] = None):

        if self.__inicializacao == "random":

            # Normal(0,1)
            self.pesos = tf.Variable(
                tf.random.normal(
                    shape=[self.__linhas, self.__colunas, self.__dimensao],
                    mean=0.0,
                    stddev=1.0,
                    dtype=tf.float32
                ),
                trainable=False,
                name="pesos_som"
            )

        elif self.__inicializacao == "sample":

            if dados is None:
                raise ValueError(
                    "Para inicialização 'sample', é necessário fornecer dados.")

            # Seleciona índices aleatórios
            n_amostras = tf.shape(dados)[0]

            indices = tf.random.uniform(
                shape=[self.__linhas, self.__colunas],
                minval=0,
                maxval=n_amostras,
                dtype=tf.int32
            )

            pesos_iniciais = tf.gather(dados, indices)

            self.pesos = tf.Variable(
                pesos_iniciais,
                trainable=False,
                name="pesos_som"
            )

        else:
            raise ValueError("Tipo de inicialização inválido.")

        return self.pesos
