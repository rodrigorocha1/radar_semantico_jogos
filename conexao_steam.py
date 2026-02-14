import pandas as pd
import s3fs
pd.set_option("display.max_columns", None)

# Não quebrar linha no meio
pd.set_option("display.expand_frame_repr", False)

# Aumentar largura do display
pd.set_option("display.width", 200)

# Mostrar texto completo (ou limite maior)
pd.set_option("display.max_colwidth", 300)

# Mostrar mais linhas
pd.set_option("display.max_rows", 20)

# Conexão com MinIO
fs = s3fs.S3FileSystem(
    key="minio",
    secret="minio123",
    client_kwargs={"endpoint_url": "http://localhost:9000"},
)

# Caminho base (sem s3:// aqui)
base_path = "extracao/steam/bronze/reviews_steam"

# Busca todos os JSON recursivamente
arquivos = fs.glob(f"{base_path}/**/*.json")

print(f"Arquivos encontrados: {len(arquivos)}")

# Lê e concatena
dfs = []

for arquivo in arquivos:
    with fs.open(arquivo) as f:
        dfs.append(pd.read_json(f, lines=True))  # 👈 geralmente Steam é JSON Lines

df : pd.DataFrame = pd.concat(dfs, ignore_index=True)



df_tratado = df[['recommendationid','codigo_steam','nome_jogo' ,'review']]
df_tratado.rename(columns={'recommendationid': 'id_texto', 'review': 'texto_comentario'}, inplace=True)

print(df_tratado.head())
print("Total registros:", len(df_tratado))
