import time

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.criacao_dataframe_completo_corrente import \
    CriacaoDataframeCompletoCorrente
from src.corrente_pipeline_comentarios.guadar_dataframe_prata import GuardarDataFramePrata
from src.corrente_pipeline_comentarios.limpeza_comentarios_corrente import \
    LimpezaComentariosCorrente
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
p2 = LimpezaComentariosCorrente()
p3 = GuardarDataFramePrata(operacoes_banco=servico_banco)

p1.set_proxima_corrente(p2).set_proxima_corrente(p3)
p1.corrente(contexto)
fim = time.perf_counter()
tempo_total = fim - inicio

print(f"\n⏱ Tempo total de execução: {tempo_total:.4f} segundos")
