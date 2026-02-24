from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Optional, Literal


class SOM(tf.keras.Model):

    def __init__(
        self,
        linhas: int,
        colunas: int,
        dimensao: int,
        taxa_aprendizado: float = 0.5,
        sigma: Optional[float] = None,
        metrica: Literal["euclidiana", "cosseno"] = "euclidiana",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.linhas = linhas
        self.colunas = colunas
        self.dimensao = dimensao
        self.total_neuronios = linhas * colunas
        self.taxa_inicial = taxa_aprendizado
        self.sigma_inicial = sigma if sigma else max(linhas, colunas) / 2
        self.metrica = metrica

        self.pesos = self.add_weight(
            shape=(self.total_neuronios, dimensao),
            initializer="random_normal",
            trainable=False,
            name="pesos_som"
        )

        self.localizacoes = self._criar_localizacoes_grid()

    # --------------------------------------------------
    # GRID
    # --------------------------------------------------

    def _criar_localizacoes_grid(self):
        x = tf.range(self.linhas, dtype=tf.float32)
        y = tf.range(self.colunas, dtype=tf.float32)
        grid = tf.stack(tf.meshgrid(x, y), axis=-1)
        return tf.reshape(grid, [-1, 2])  # (N,2)

    # --------------------------------------------------
    # DISTÂNCIA (TOTALMENTE VETORIZADA)
    # --------------------------------------------------

    def _distancias(self, X):
        # X: (B, D)
        # pesos: (N, D)

        if self.metrica == "cosseno":
            X = tf.nn.l2_normalize(X, axis=1)
            W = tf.nn.l2_normalize(self.pesos, axis=1)
            return 1 - tf.matmul(X, W, transpose_b=True)

        # ||x-w||² = ||x||² + ||w||² - 2xwᵀ
        x2 = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)  # (B,1)
        w2 = tf.reduce_sum(tf.square(self.pesos), axis=1)        # (N,)
        return x2 + w2 - 2 * tf.matmul(X, self.pesos, transpose_b=True)

    # --------------------------------------------------
    # PASSO VETORIZADO
    # --------------------------------------------------

    @tf.function
    def train_step_som(self, X, passo_atual, total_passos):

        taxa = self.taxa_inicial * tf.exp(-passo_atual / total_passos)
        sigma = self.sigma_inicial * tf.exp(-passo_atual / total_passos)

        # 1️⃣ Distância entrada ↔ neurônios
        distancias = self._distancias(X)  # (B, N)

        # 2️⃣ BMU por amostra
        bmu_idx = tf.argmin(distancias, axis=1)  # (B,)

        # 3️⃣ Coordenadas dos BMUs
        bmu_loc = tf.gather(self.localizacoes, bmu_idx)  # (B,2)

        # 4️⃣ Distância de grade vetorizada
        loc_expand = tf.expand_dims(self.localizacoes, 0)   # (1,N,2)
        bmu_expand = tf.expand_dims(bmu_loc, 1)             # (B,1,2)

        dist_grade = tf.reduce_sum(
            tf.square(loc_expand - bmu_expand),
            axis=2
        )  # (B,N)

        # 5️⃣ Função vizinhança gaussiana
        h = tf.exp(-dist_grade / (2.0 * sigma ** 2))  # (B,N)

        # 6️⃣ Atualização vetorizada real
        # Expandir dimensões
        h = tf.expand_dims(h, -1)        # (B,N,1)
        X_expand = tf.expand_dims(X, 1)  # (B,1,D)
        W_expand = tf.expand_dims(self.pesos, 0)  # (1,N,D)

        delta = taxa * h * (X_expand - W_expand)  # (B,N,D)

        # Soma ao invés de média (mais fiel ao batch learning)
        delta_total = tf.reduce_mean(delta, axis=0)  # (N,D)

        self.pesos.assign_add(delta_total)

        return taxa, sigma