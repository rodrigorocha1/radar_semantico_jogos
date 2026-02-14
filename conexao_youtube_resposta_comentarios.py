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
base_path = "extracao/youtube/bronze/resposta_comentarios_youtube"

# Busca todos os JSON recursivamente
arquivos = fs.glob(f"{base_path}/**/*.json")

print(f"Arquivos encontrados: {len(arquivos)}")

# Lê e concatena
dfs = []

for arquivo in arquivos:
    with fs.open(arquivo) as f:
        dfs.append(pd.read_json(f, lines=True))  # 👈 geralmente Steam é JSON Lines

df : pd.DataFrame = pd.concat(dfs, ignore_index=True)
jogos_dict_invertido = {
    "star_rupture": 1631270,
    "no_mans_sky": 275850,
    "x4_foundations": 392160,
    "satisfactory": 526870,
    "planet_crafter": 1284190,
    "the_planet_crafter_toxicity": 4078590,
    "the_planet_crafter_planet_humble": 3142540,
    "elite_dangerous": 359320,
    "elite_dangerous_odissey": 1336350,
    "euro_truck_simulator": 227300,
    "euro_truck_simulator_grecia": 2604420,
    "euro_truck_simulator_iberia": 1209460,
    "euro_truck_simulator_italia": 558244,
    "euro_truck_simulator_beyound_the_baltic_sea": 925580,
    "euro_truck_simulator_road_to_the_black_sea": 1056760,
    "euro_truck_simulator_vive_le_france": 531130,
    "euro_truck_simulator_scandinaavia": 304212,
    "euro_truck_simulator_going_east": 227310,
    "euro_truck_simulator_nordic_horizons": 2780810,
    "space_engineers": 244850,
    "cities_skylines": 255710,
    "cities_skylines_dois": 949230,
}

mapa_codigos = dict(jogos_dict_invertido)

# adiciona coluna
df["codigo_steam"] = df["nome_jogo"].map(mapa_codigos)

print(df['codigo_steam'].head())

df["textDisplay"] = df["snippet"].apply(
    lambda x: x["topLevelComment"]["snippet"]["textDisplay"]
)
# print(df.head())
# df.rename(columns={'id' : 'id_texto'},  inplace=True)

df_final = df[['id','codigo_steam', 'nome_jogo', 'textDisplay']]
df_final.rename(columns={'id' : 'id_texto', 'textDisplay': 'texto_comentario'},  inplace=True)

print(df_final.head())

# df_tratado = df[['recommendationid','codigo_steam','nome_jogo' ,'review']]
#
