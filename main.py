from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.obter_comentarios_steam_corrente import ObterComentariosSteamCorrente
from src.corrente_pipeline_comentarios.verificar_conexao_api_steam_corrente import VerificarConexaoApiSteamCorrente
from src.servicos.steam.steam_api import SteamAPI

contexto = Contexto()
lista_jogos = []
steam_api = SteamAPI()

p1 = VerificarConexaoApiSteamCorrente(steam_api=steam_api)
p2 = ObterComentariosSteamCorrente(api_steam=SteamAPI(), lista_jogos=lista_jogos)
p1.set_proxima_corrente(p2)
p1.corrente(contexto=contexto)
