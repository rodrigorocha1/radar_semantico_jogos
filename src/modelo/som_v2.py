from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class SOMV2(tf.Module):

    def __init__(
        self,
        linhas: int,
        colunas: int,
        dimensao: int,
        taxa_aprendizado: float = 0.5,
        sigma: float = 1.0,
        metrica: Literal["euclidiana", "cosseno"] = "euclidiana",
    ):

        self.__linhas = linhas
        self.__colunas = colunas
        self.__dimensao = dimensao
        self.__total_neuronios = linhas * colunas

        self.__taxa_aprendizado = taxa_aprendizado
        self.__sigma = sigma
        self.__metrica = metrica

        self.__global_step = 0
        self.historico_pesos = []

        # -----------------------------
        # Inicializa grid de coordenadas
        # -----------------------------
        lin = tf.range(self.__linhas, dtype=tf.float32)
        col = tf.range(self.__colunas, dtype=tf.float32)
        grid_l, grid_c = tf.meshgrid(lin, col, indexing="ij")

        self.localizacoes = tf.reshape(
            tf.stack([grid_l, grid_c], axis=-1),
            [-1, 2]
        )  # (total_neuronios, 2)

        # -----------------------------
        # Inicializa pesos
        # -----------------------------
        self.pesos = tf.Variable(
            tf.random.normal(
                shape=[self.__linhas, self.__colunas, self.__dimensao],
                mean=0.0,
                stddev=1.0,
                dtype=tf.float32
            ),
            trainable=False
        )

    # ==========================================================
    # Distância
    # ==========================================================

    def _calcular_distancia(self, entrada, pesos):

        if self.__metrica == "cosseno":
            entrada = tf.nn.l2_normalize(entrada, axis=-1)
            pesos = tf.nn.l2_normalize(pesos, axis=-1)
            similaridade = tf.reduce_sum(entrada * pesos, axis=-1)
            return 1.0 - similaridade

        return tf.reduce_sum(
            tf.square(entrada - pesos),
            axis=-1
        )

    # ==========================================================
    # Passo de treinamento vetorizado
    # ==========================================================

    @tf.function
    def passo_treinamento(
        self,
        entrada: tf.Tensor,
        passo_atual: tf.Tensor,
        total_passos: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        taxa = self.__taxa_aprendizado * tf.exp(
            -passo_atual / total_passos
        )

        sigma = self.__sigma * tf.exp(
            -passo_atual / total_passos
        )

        # -----------------------------------
        # Broadcasting correto
        # -----------------------------------
        entrada_expandida = tf.expand_dims(
            tf.expand_dims(entrada, 1), 1
        )  # (batch,1,1,dim)

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )  # (1,lin,col,dim)

        # -----------------------------------
        # Distância
        # -----------------------------------
        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )  # (batch, lin, col)

        # -----------------------------------
        # BMU correto
        # -----------------------------------
        dist_flat = tf.reshape(
            distancias,
            [tf.shape(distancias)[0], -1]
        )

        indice_bmu = tf.argmin(dist_flat, axis=1)

        local_bmu = tf.gather(
            self.localizacoes,
            indice_bmu
        )  # (batch,2)

        # -----------------------------------
        # Distância na grade
        # -----------------------------------
        loc_expand = tf.expand_dims(
            self.localizacoes, 0
        )  # (1,total,2)

        bmu_expand = tf.expand_dims(
            local_bmu, 1
        )  # (batch,1,2)

        dist_grade = tf.reduce_sum(
            tf.square(loc_expand - bmu_expand),
            axis=2
        )  # (batch,total)

        vizinhanca = tf.exp(
            -dist_grade / (2.0 * sigma ** 2)
        )  # (batch,total)

        vizinhanca = tf.reshape(
            vizinhanca,
            [-1, self.__linhas, self.__colunas, 1]
        )

        # -----------------------------------
        # Atualização vetorizada
        # -----------------------------------
        delta = taxa * vizinhanca * (
            entrada_expandida - pesos_expandidos
        )

        novo_peso = self.pesos + tf.reduce_mean(delta, axis=0)

        self.pesos.assign(novo_peso)

        return taxa, sigma

    # ==========================================================
    # Treinamento
    # ==========================================================

    def treinar(
        self,
        dataset: tf.data.Dataset,
        epocas: int
    ):

        # -------------------------------------------------
        # Reconstrói dataset completo para métricas
        # -------------------------------------------------
        X_completo = tf.concat([x for x in dataset], axis=0)

        num_batches = sum(1 for _ in dataset)
        total_passos = tf.constant(
            epocas * num_batches,
            dtype=tf.float32
        )

        # Histórico opcional
        self.historico_qe = []
        self.historico_te = []

        print("\nIniciando treinamento do SOM")
        print(f"Grid: {self.__linhas}x{self.__colunas}")
        print(f"Dimensão vetorial: {self.__dimensao}")
        print(f"Total neurônios: {self.__total_neuronios}")
        print(f"Épocas: {epocas}")
        print(f"Batches por época: {num_batches}")
        print("-" * 50)

        for epoca in range(epocas):

            for lote in tqdm(dataset, leave=False):

                taxa, sigma = self.passo_treinamento(
                    lote,
                    tf.constant(self.__global_step, dtype=tf.float32),
                    total_passos
                )

                self.__global_step += 1

            # -------------------------------------------------
            # Métricas após cada época
            # -------------------------------------------------
            qe = self.calcular_erro_quantizacao(X_completo)
            te = self.calcular_erro_topografico(X_completo)

            self.historico_qe.append(qe.numpy())
            self.historico_te.append(te.numpy())

            print(
                f"Época {epoca+1}/{epocas} concluída | "
                f"QE: {qe.numpy():.6f} | "
                f"TE: {te.numpy():.6f}"
            )

        print("\nTreinamento finalizado.")

    def obter_mapa_ativacao(self, entrada: tf.Tensor) -> tf.Tensor:
        """
        Retorna o mapa de ativação para cada amostra.

        Entrada:
            (batch, dim)

        Saída:
            (batch, linhas, colunas)
        """

        if len(entrada.shape) == 1:
            entrada = tf.expand_dims(entrada, 0)

        entrada_expandida = tf.expand_dims(
            tf.expand_dims(entrada, 1), 1
        )  # (batch,1,1,dim)

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )  # (1,lin,col,dim)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )

        return distancias

    def obter_resposta_ativacao(self, X: tf.Tensor) -> tf.Tensor:
        """
        Equivalente ao MiniSom.activation_response(X)
        Um grid (linhas, colunas) onde cada célula contém quantas vezes aquele neurônio foi BMU para o conjunto X.

        Entrada:
            X -> (N, dim)

        Saída:
            (linhas, colunas) contendo contagem de BMUs
        """

        # Garante batch
        if len(X.shape) == 1:
            X = tf.expand_dims(X, 0)

        # -----------------------------
        # Calcula mapa de ativação
        # -----------------------------
        entrada_expandida = tf.expand_dims(
            tf.expand_dims(X, 1), 1
        )  # (N,1,1,dim)

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )  # (1,lin,col,dim)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )  # (N,lin,col)

        # -----------------------------
        # Flatten para encontrar BMU
        # -----------------------------
        dist_flat = tf.reshape(
            distancias,
            [tf.shape(distancias)[0], -1]
        )

        indice_bmu = tf.argmin(dist_flat, axis=1)  # (N,)

        # -----------------------------
        # Conta frequência
        # -----------------------------
        contagem = tf.math.bincount(
            indice_bmu,
            minlength=self.__total_neuronios,
            maxlength=self.__total_neuronios,
            dtype=tf.int32
        )

        # -----------------------------
        # Volta para formato grid
        # -----------------------------
        response = tf.reshape(
            contagem,
            [self.__linhas, self.__colunas]
        )

        return response

    def distance_map(self, scaling: str = "sum") -> tf.Tensor:
        """
        Calcula a U-Matrix (distance map).

        Cada célula contém a soma (ou média) das distâncias
        euclidianas entre um neurônio e seus vizinhos imediatos.

        Parâmetros
        ----------
        scaling : "sum" | "mean"
            - "sum"  -> soma das distâncias
            - "mean" -> média das distâncias

        Retorno
        -------
        (linhas, colunas) normalizado entre 0 e 1
        """

        if scaling not in ("sum", "mean"):
            raise ValueError('scaling deve ser "sum" ou "mean"')

        pesos = self.pesos  # (lin, col, dim)

        # -------------------------------------------------
        # Diferenças horizontais
        # -------------------------------------------------
        diff_h = pesos[:, 1:, :] - pesos[:, :-1, :]
        dist_h = tf.norm(diff_h, axis=-1)

        # Padding para manter shape original
        dist_h = tf.pad(dist_h, [[0, 0], [0, 1]])

        # -------------------------------------------------
        # Diferenças verticais
        # -------------------------------------------------
        diff_v = pesos[1:, :, :] - pesos[:-1, :, :]
        dist_v = tf.norm(diff_v, axis=-1)

        dist_v = tf.pad(dist_v, [[0, 1], [0, 0]])

        # -------------------------------------------------
        # Diferenças diagonais (opcional – 8 vizinhos como MiniSom)
        # -------------------------------------------------
        diff_d1 = pesos[1:, 1:, :] - pesos[:-1, :-1, :]
        dist_d1 = tf.norm(diff_d1, axis=-1)
        dist_d1 = tf.pad(dist_d1, [[0, 1], [0, 1]])

        diff_d2 = pesos[1:, :-1, :] - pesos[:-1, 1:, :]
        dist_d2 = tf.norm(diff_d2, axis=-1)
        dist_d2 = tf.pad(dist_d2, [[0, 1], [1, 0]])

        # -------------------------------------------------
        # Soma todas as contribuições
        # -------------------------------------------------
        u_matrix = dist_h + dist_v + dist_d1 + dist_d2

        if scaling == "mean":
            # Número máximo de vizinhos = 4 (hv) + 2 diagonais = 6
            # bordas terão menos vizinhos — aproximamos dividindo por 6
            u_matrix = u_matrix / 6.0

        # -------------------------------------------------
        # Normalização 0-1
        # -------------------------------------------------
        max_val = tf.reduce_max(u_matrix)
        u_matrix = tf.where(max_val > 0, u_matrix / max_val, u_matrix)

        return u_matrix

    @tf.function
    def obter_neuronio_vencedor(self, entrada: tf.Tensor) -> tf.Tensor:

        entrada = tf.reshape(entrada, [-1])

        entrada_expandida = tf.reshape(
            entrada,
            [1, 1, self.__dimensao]
        )

        distancias = self._calcular_distancia(
            entrada_expandida,
            self.pesos
        )

        indice_flat = tf.argmin(
            tf.reshape(distancias, [-1]),
            output_type=tf.int32   # ← FIX AQUI
        )

        coords = tf.unravel_index(
            indice_flat,
            tf.constant(
                [self.__linhas, self.__colunas],
                dtype=tf.int32
            )
        )

        return tf.stack(coords)

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

            # if len(docs) < min_docs:
            #     continue

            centroide = emb_cluster.mean(axis=0)

            distancias = np.linalg.norm(
                emb_cluster - centroide,
                axis=1
            )

            idx_representativo = np.argmin(distancias)

            rotulos[int(neuronio)] = docs[idx_representativo]

        return rotulos

    def calcular_erro_quantizacao(self, X: tf.Tensor) -> tf.Tensor:
        """
        Calcula o erro médio de quantização.

        QE = média da distância entre cada amostra
            e seu neurônio vencedor (BMU).
        """

        if len(X.shape) == 1:
            X = tf.expand_dims(X, 0)

        # Expande para broadcasting
        entrada_expandida = tf.expand_dims(
            tf.expand_dims(X, 1), 1
        )  # (N,1,1,dim)

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )  # (1,lin,col,dim)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )  # (N,lin,col)

        dist_flat = tf.reshape(
            distancias,
            [tf.shape(distancias)[0], -1]
        )

        # distância do BMU
        min_dist = tf.reduce_min(dist_flat, axis=1)

        return tf.reduce_mean(min_dist)

    def calcular_erro_topografico(self, X: tf.Tensor) -> tf.Tensor:
        """
        Calcula o erro topológico.

        Mede a proporção de amostras cujo
        1º e 2º BMUs não são vizinhos na grade.
        """

        if len(X.shape) == 1:
            X = tf.expand_dims(X, 0)

        N = tf.shape(X)[0]

        entrada_expandida = tf.expand_dims(
            tf.expand_dims(X, 1), 1
        )

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )  # (N,lin,col)

        dist_flat = tf.reshape(
            distancias,
            [N, -1]
        )

        # Ordena distâncias
        valores, indices = tf.math.top_k(
            -dist_flat,  # negativo para pegar menores
            k=2
        )

        bmu1 = indices[:, 0]
        bmu2 = indices[:, 1]

        # Converte para coordenadas
        coords1 = tf.stack(
            tf.unravel_index(
                bmu1,
                (self.__linhas, self.__colunas)
            ),
            axis=1
        )

        coords2 = tf.stack(
            tf.unravel_index(
                bmu2,
                (self.__linhas, self.__colunas)
            ),
            axis=1
        )

        # Distância na grade
        dist_grid = tf.reduce_sum(
            tf.abs(coords1 - coords2),
            axis=1
        )

        # São vizinhos se distância Manhattan == 1
        erro = tf.cast(dist_grid > 1, tf.float32)

        return tf.reduce_mean(erro)

    def mapear(
        self,
        X: np.ndarray
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)


        entrada_expandida = tf.expand_dims(
            tf.expand_dims(X_tensor, 1), 1
        )  # (N,1,1,dim)

        pesos_expandidos = tf.expand_dims(
            self.pesos, 0
        )  # (1,lin,col,dim)

        distancias = self._calcular_distancia(
            entrada_expandida,
            pesos_expandidos
        )  # (N,lin,col)

        dist_flat = tf.reshape(
            distancias,
            [tf.shape(distancias)[0], -1]
        )

        indices_bmu = tf.argmin(
            dist_flat,
            axis=1,
            output_type=tf.int32
        )

        coordenadas = tf.gather(
            self.localizacoes,
            indices_bmu
        )

        return indices_bmu, coordenadas
