import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.servico_s3.iservicos3 import Iservicos3


class CriacaoDataframeCompletoCorrente(Corrente):

    def __init__(self, servico_s3: Iservicos3):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__jogos_dict_invertido = {
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

    def __criar_dataframe_steam(self) -> pd.DataFrame:
        caminho_base = "extracao/steam/bronze/reviews_steam"
        dataframe = self.__servico_s3.ler_jsons_para_dataframe(caminho_base)
        dataframe = dataframe[['recommendationid', 'codigo_steam', 'nome_jogo', 'review']]
        dataframe.rename(columns={'recommendationid': 'id_texto', 'review': 'texto_comentario'}, inplace=True)
        return dataframe

    def executar_processo(self, contexto: Contexto) -> bool:
        dataframe_steam = self.__criar_dataframe_steam()

        return True
