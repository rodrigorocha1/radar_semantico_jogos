from __future__ import annotations

import datetime
import os
from typing import Dict, List, Literal, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


class SOM(tf.Module):

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
    ) -> None:
        super().__init__()

        self.linhas = linhas
        self.colunas = colunas
        self.dimensao = dimensao
        self.taxa_aprendizado = taxa_aprendizado
        self.sigma = sigma if sigma else max(linhas, colunas) / 2
        self.metrica = metrica
        self.inicializacao = inicializacao
        self.total_neuronios = linhas * colunas

        self.pesos = tf.Variable(
            tf.random.normal([self.total_neuronios, dimensao]),
            trainable=False
        )

        self.localizacoes = self._criar_localizacoes_grid()

        # -------- TensorBoard --------
        if log_dir is None:
            log_dir = os.path.join(
                "logs",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        self.writer = tf.summary.create_file_writer(log_dir)
        self.global_step = 0

    # ---------------------------------------------------
    # GRID
    # ---------------------------------------------------

    def _criar_localizacoes_grid(self) -> tf.Tensor:
        eixo_x = tf.range(self.linhas, dtype=tf.float32)
        eixo_y = tf.range(self.colunas, dtype=tf.float32)
        grade = tf.stack(tf.meshgrid(eixo_x, eixo_y), axis=-1)
        return tf.reshape(grade, [-1, 2])

    # ---------------------------------------------------
    # MÉTRICA
    # ---------------------------------------------------

    def _calcular_distancia(self, entrada_expandida, pesos_expandidos):

        if self.metrica == "cosseno":
            entrada_norm = tf.nn.l2_normalize(entrada_expandida, axis=2)
            pesos_norm = tf.nn.l2_normalize(pesos_expandidos, axis=2)
            similaridade = tf.reduce_sum(
                entrada_norm * pesos_norm, axis=2
            )
            return 1 - similaridade

        return tf.reduce_sum(
            tf.square(entrada_expandida - pesos_expandidos),
            axis=2
        )

    # ---------------------------------------------------
    # PASSO TREINAMENTO (sem summary)
    # ---------------------------------------------------

    @tf.function
    def passo_treinamento(
        self,
        entrada,
        passo_atual,
        total_passos
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        taxa = self.taxa_aprendizado * tf.exp(
            -passo_atual / total_passos
        )

        sigma = self.sigma * tf.exp(
            -passo_atual / total_passos
        )

        entrada_expandida = tf.expand_dims(entrada, 1)
        pesos_expandidos = tf.expand_dims(self.pesos, 0)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )

        indice_bmu = tf.argmin(distancias, axis=1)
        localizacoes_bmu = tf.gather(
            self.localizacoes,
            indice_bmu
        )

        localizacoes_expandidas = tf.expand_dims(
            self.localizacoes, 0
        )
        bmu_expandida = tf.expand_dims(
            localizacoes_bmu, 1
        )

        distancia_grade = tf.reduce_sum(
            tf.square(localizacoes_expandidas - bmu_expandida),
            axis=2
        )

        vizinhanca = tf.exp(
            -distancia_grade / (2 * sigma ** 2)
        )

        influencia = tf.expand_dims(vizinhanca, -1)

        delta = taxa * influencia * (
            entrada_expandida - pesos_expandidos
        )

        novos_pesos = self.pesos + tf.reduce_mean(delta, axis=0)
        self.pesos.assign(novos_pesos)

        return taxa, sigma

    # ---------------------------------------------------
    # TREINAMENTO COM TENSORBOARD
    # ---------------------------------------------------

    def treinar(
        self,
        dataset: tf.data.Dataset,
        epocas: int,
        calcular_erro: bool = False,
        dados_completos: Optional[np.ndarray] = None
    ) -> None:

        # Conta batches sem materializar dados
        num_batches = sum(1 for _ in dataset)
        total_passos = tf.constant(
            epocas * num_batches,
            dtype=tf.float32
        )

        print("\n🚀 Iniciando treinamento do SOM")
        print(f"📌 Grid: {self.linhas}x{self.colunas}")
        print(f"📌 Dimensão vetorial: {self.dimensao}")
        print(f"📌 Total neurônios: {self.total_neuronios}")
        print(f"📌 Épocas: {epocas}")
        print(f"📌 Batches por época: {num_batches}")
        print("-" * 50)

        for epoca in range(epocas):

            barra = tqdm(
                dataset,
                total=num_batches,
                desc=f"{'-' * 10} ->Época {epoca+1}/{epocas}<-{'-' * 10}",
                leave=False
            )

            for lote in barra:

                taxa, sigma = self.passo_treinamento(
                    lote,
                    tf.constant(self.global_step, dtype=tf.float32),
                    total_passos
                )

                with self.writer.as_default():
                    tf.summary.scalar(
                        "taxa_aprendizado",
                        taxa,
                        step=self.global_step
                    )
                    tf.summary.scalar(
                        "sigma",
                        sigma,
                        step=self.global_step
                    )
                    tf.summary.histogram(
                        "pesos",
                        self.pesos,
                        step=self.global_step
                    )

                # Atualiza descrição dinâmica da barra
                barra.set_postfix({
                    "taxa": f"{float(taxa.numpy()):.4f}",
                    "sigma": f"{float(sigma.numpy()):.4f}"
                })

                self.global_step += 1

            # -------- Erro de Quantização --------
            if calcular_erro and dados_completos is not None:
                erro = self.erro_quantizacao(dados_completos)

                with self.writer.as_default():
                    tf.summary.scalar(
                        "erro_quantizacao",
                        erro,
                        step=epoca
                    )

                print(
                    f" ->  Época {epoca+1}: Erro de quantização = {erro:.6f} <-")

            # -------- U-Matrix --------
            u_matrix = self.calcular_u_matrix()
            u_img = np.expand_dims(u_matrix, axis=(0, -1))

            with self.writer.as_default():
                tf.summary.image(
                    "U_Matrix",
                    u_img,
                    step=epoca
                )

            print(f" Época {epoca+1} concluída.")

        self.writer.flush()

        print("\nTreinamento finalizado com sucesso.")

    # ---------------------------------------------------
    # ERRO QUANTIZAÇÃO
    # ---------------------------------------------------

    def erro_quantizacao(self, X) -> float:
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        entrada_expandida = tf.expand_dims(X_tensor, 1)
        pesos_expandidos = tf.expand_dims(self.pesos, 0)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )

        menor_distancia = tf.reduce_min(
            distancias,
            axis=1
        )

        return float(tf.reduce_mean(menor_distancia).numpy())

    # ---------------------------------------------------
    # U-MATRIX
    # ---------------------------------------------------

    def calcular_u_matrix(self) -> np.ndarray:
        """
        Calcula a U-Matrix de forma vetorizada.
        Usa vizinhança 4-direcional.
        """

        pesos = self.pesos.numpy().reshape(
            self.linhas,
            self.colunas,
            self.dimensao
        )

        u_matrix = np.zeros((self.linhas, self.colunas))
        contador = np.zeros((self.linhas, self.colunas))

        # -------------------------------------------------
        # Distância com vizinho de baixo
        # -------------------------------------------------
        diff_down = np.linalg.norm(
            pesos[:-1, :, :] - pesos[1:, :, :],
            axis=2
        )

        u_matrix[:-1, :] += diff_down
        u_matrix[1:, :] += diff_down

        contador[:-1, :] += 1
        contador[1:, :] += 1

        # -------------------------------------------------
        # Distância com vizinho da direita
        # -------------------------------------------------
        diff_right = np.linalg.norm(
            pesos[:, :-1, :] - pesos[:, 1:, :],
            axis=2
        )

        u_matrix[:, :-1] += diff_right
        u_matrix[:, 1:] += diff_right

        contador[:, :-1] += 1
        contador[:, 1:] += 1

        # -------------------------------------------------
        # Média final
        # -------------------------------------------------
        u_matrix /= contador

        return u_matrix

    # ---------------------------------------------------
# U-MATRIX VETORIZADA (ALTA PERFORMANCE)
# ---------------------------------------------------

    def rotular_por_centroide(
        self,
        textos: List[str],
        embeddings: np.ndarray,
        min_docs: int = 3
    ) -> Dict[int, str]:

        indices, _ = self.mapear(embeddings)
        indices = indices.numpy()

        rotulos: Dict[int, str] = {}

        for neuronio in np.unique(indices):

            mask = indices == neuronio
            docs = np.array(textos)[mask]
            emb_cluster = embeddings[mask]

            if len(docs) < min_docs:
                continue

            centroide = emb_cluster.mean(axis=0)

            distancias = np.linalg.norm(
                emb_cluster - centroide,
                axis=1
            )

            idx_representativo = np.argmin(distancias)

            rotulos[int(neuronio)] = docs[idx_representativo]

        return rotulos

    def mapear(
        self,
        X: np.ndarray
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        entrada_expandida = tf.expand_dims(X_tensor, 1)
        pesos_expandidos = tf.expand_dims(self.pesos, 0)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )

        indices_bmu = tf.argmin(distancias, axis=1)
        coordenadas = tf.gather(self.localizacoes, indices_bmu)

        return indices_bmu, coordenadas

    def plotar_decay(self, num_epocas: int, num_batches: int):
        """
        Plota o decaimento da taxa de aprendizado e do sigma ao longo do treinamento.
        """
        total_passos = num_epocas * num_batches
        passos = np.arange(total_passos)

        # Decaimento exponencial
        taxa = self.taxa_aprendizado * np.exp(-passos / total_passos)
        sigma = self.sigma * np.exp(-passos / total_passos)

        plt.figure(figsize=(10, 5))
        plt.plot(passos, taxa, label='Taxa de Aprendizado')
        plt.plot(passos, sigma, label='Sigma (Vizinhança)')
        plt.xlabel('Passos de Treinamento')
        plt.ylabel('Valor')
        plt.title('Decaimento da Taxa de Aprendizado e Sigma do SOM')
        plt.grid(True)
        plt.legend()
        plt.show()
