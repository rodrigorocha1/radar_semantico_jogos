from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.guardar_dados_steam_s3_corrente import GuardarDadosSteam3Corrente
from src.corrente_pipeline_comentarios.obter_comentarios_steam_corrente import ObterComentariosSteamCorrente
from src.corrente_pipeline_comentarios.verificar_conexao_api_steam_corrente import VerificarConexaoApiSteamCorrente
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb
from src.servicos.servico_s3.sevicos3 import ServicoS3
from src.servicos.steam.steam_api import SteamAPI

contexto = Contexto(gerador_reviews_steam=None)
lista_jogos = [1631270, 275850]
steam_api = SteamAPI()
servico_s3 = ServicoS3()
servico_banco = OperacoesBancoDuckDb()
p1 = VerificarConexaoApiSteamCorrente(steam_api=steam_api)
p2 = ObterComentariosSteamCorrente(api_steam=SteamAPI(), lista_jogos=lista_jogos)
p3 = GuardarDadosSteam3Corrente(
    servico_s3=servico_s3,
    servico_banco=servico_banco
)
p1.set_proxima_corrente(p2).set_proxima_corrente(p3)
p1.corrente(contexto=contexto)
