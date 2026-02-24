from typing import List, Literal, Optional

import numpy as np
import tensorflow as tf


class SOMKeras(tf.keras.Model):
    def __init__(
        self,
        linhas: int,
        colunas: int,
        dimensao: int,
        taxa_aprendizado: float = 0.5,
        sigma: Optional[float] = None,
        metrica: Literal["euclidiana", "cosseno"] = "euclidiana",
        log_interval: int = 50,  # frequência do log
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
        self.log_interval = log_interval

        # Pesos do SOM
        self.pesos = self.add_weight(
            shape=(self.total_neuronios, dimensao),
            initializer="random_normal",
            trainable=False,
            name="pesos_som"
        )

        self.localizacoes = self._criar_localizacoes_grid()
        self.embedding: Optional[np.ndarray] = None

    # --------------------------
    # GRID
    # --------------------------
    def _criar_localizacoes_grid(self):
        x = tf.range(self.linhas, dtype=tf.float32)
        y = tf.range(self.colunas, dtype=tf.float32)
        grid = tf.stack(tf.meshgrid(x, y), axis=-1)
        return tf.reshape(grid, [-1, 2])

    # --------------------------
    # DISTÂNCIAS
    # --------------------------
    def _distancias(self, X):
        if self.metrica == "cosseno":
            X = tf.nn.l2_normalize(X, axis=1)
            W = tf.nn.l2_normalize(self.pesos, axis=1)
            return 1 - tf.matmul(X, W, transpose_b=True)
        x2 = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
        w2 = tf.reduce_sum(tf.square(self.pesos), axis=1)
        return x2 + w2 - 2 * tf.matmul(X, self.pesos, transpose_b=True)

    # --------------------------
    # PASSO DE TREINO
    # --------------------------
    @tf.function
    def train_step_som(self, X, passo_atual, total_passos):
        taxa = self.taxa_inicial * tf.exp(-passo_atual / total_passos)
        sigma = self.sigma_inicial * tf.exp(-passo_atual / total_passos)

        distancias = self._distancias(X)
        bmu_idx = tf.argmin(distancias, axis=1)
        bmu_loc = tf.gather(self.localizacoes, bmu_idx)

        loc_expand = tf.expand_dims(self.localizacoes, 0)
        bmu_expand = tf.expand_dims(bmu_loc, 1)
        dist_grade = tf.reduce_sum(tf.square(loc_expand - bmu_expand), axis=2)

        h = tf.exp(-dist_grade / (2.0 * sigma ** 2))
        h = tf.expand_dims(h, -1)
        X_expand = tf.expand_dims(X, 1)
        W_expand = tf.expand_dims(self.pesos, 0)
        delta = taxa * h * (X_expand - W_expand)
        delta_total = tf.reduce_mean(delta, axis=0)
        self.pesos.assign_add(delta_total)
        return taxa, sigma

    # --------------------------
    # FIT COM EMBEDDINGS
    # --------------------------
    def fit_embeddings(self, embeddings: np.ndarray, passos: int = 1000, batch_size: int = 32):
        """
        Treina o SOM com embeddings já tratados e calcula erro de quantização.
        """
        self.embedding = embeddings
        X = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        n_amostras = X.shape[0]

        for passo_atual in range(passos):
            idx = np.random.choice(n_amostras, size=batch_size, replace=False)
            batch = tf.gather(X, idx)
            taxa, sigma = self.train_step_som(batch, passo_atual, passos)

            # log a cada log_interval
            if (passo_atual + 1) % self.log_interval == 0 or passo_atual == 0:
                q_error = self.quantization_error(X)
                print(
                    f"[Passo {passo_atual+1}/{passos}] taxa={taxa.numpy():.4f}, sigma={sigma.numpy():.4f}, erro_quantizacao={q_error:.4f}")

    # --------------------------
    # ERRO DE QUANTIZAÇÃO
    # --------------------------
    def quantization_error(self, X: Optional[tf.Tensor] = None):
        """
        Retorna o erro de quantização médio: distância média entre cada vetor e seu BMU
        """
        if X is None:
            if self.embedding is None:
                raise ValueError(
                    "Nenhum embedding disponível para cálculo do erro de quantização.")
            X = tf.convert_to_tensor(self.embedding, dtype=tf.float32)

        distancias = self._distancias(X)
        min_dist = tf.reduce_min(distancias, axis=1)
        return tf.reduce_mean(min_dist).numpy()

    # --------------------------
    # MAPEAR COMENTÁRIOS
    # --------------------------
    def map_embeddings(self, embeddings: Optional[np.ndarray] = None):
        if embeddings is None:
            if self.embedding is None:
                raise ValueError(
                    "Nenhum embedding disponível. Passe embeddings ou treine primeiro.")
            embeddings = self.embedding

        X = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        distancias = self._distancias(X)
        bmu_idx = tf.argmin(distancias, axis=1)
        bmu_loc = tf.gather(self.localizacoes, bmu_idx)
        return bmu_loc.numpy()

    # --------------------------
    # GERAR MAPA SEMÂNTICO
    # --------------------------
    def get_grid_map(self, textos: List[str], embeddings: Optional[np.ndarray] = None):
        locs = self.map_embeddings(embeddings)
        mapa = {}
        for texto, loc in zip(textos, locs):
            key = tuple(map(int, loc))
            if key not in mapa:
                mapa[key] = []
            mapa[key].append(texto)
        return mapa
