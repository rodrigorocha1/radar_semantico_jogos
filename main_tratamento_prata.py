import time

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.criacao_dataframe_completo_corrente import CriacaoDataframeCompletoCorrente
from src.corrente_pipeline_comentarios.guardar_dados_comentarios_youtube_s3_corrente import \
    GuardarDadosYoutubeComentariosS3Corrente
from src.corrente_pipeline_comentarios.guardar_dados_resposta_comentarios_youtube_s3_corrente import \
    GuardarDadosYoutubeRespostaComentariosS3Corrente
from src.corrente_pipeline_comentarios.guardar_dados_steam_s3_corrente import GuardarDadosSteam3Corrente
from src.corrente_pipeline_comentarios.obter_comentarios_steam_corrente import ObterComentariosSteamCorrente
from src.corrente_pipeline_comentarios.obter_comentarios_youtube_corrente import ObterComentariosYoutubeCorrente
from src.corrente_pipeline_comentarios.obter_resposta_comentarios_youtube_corrente import \
    ObterRespostaComentariosYoutubeCorrente
from src.corrente_pipeline_comentarios.verificar_conexao_api_steam_corrente import VerificarConexaoApiSteamCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb
from src.servicos.servico_s3.sevicos3 import ServicoS3
from src.servicos.steam.steam_api import SteamAPI

inicio = time.perf_counter()


contexto = Contexto()

steam_api = SteamAPI()
servico_s3 = ServicoS3()
servico_banco = OperacoesBancoDuckDb()
api_youtube = YoutubeAPI()

p1 = CriacaoDataframeCompletoCorrente(servico_banco=servico_banco)

p1.corrente(contexto)
fim = time.perf_counter()
tempo_total = fim - inicio

print(f"\n⏱ Tempo total de execução: {tempo_total:.4f} segundos")
