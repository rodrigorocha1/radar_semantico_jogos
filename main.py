from src.contexto.contexto import Contexto
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

'''
1631270 - Star Rupture
275850 - No Man's Sky
392160 - X4: Foundations
526870 - Satisfactory
1284190 - Planet Crafter
4078590- The Planet Crafter - Toxicity
3142540 - The Planet Crafter - Planet Humble
359320 - Elite Dangerous
1336350 -  Elite Dangerous: Odyssey

'''
contexto = Contexto()
lista_jogos = [1631270, 275850, 392160, 526870, 1284190, 4078590, 3142540, 359320, 1336350]
lista_jogos_youtube = [
    ('WLilIKOJYi0', 'star_rupture'),
    ('D1PDHTGNswI', 'star_rupture'),
    ('9Ib27hgkG3s', 'star_rupture'),
    ('OvHk_itOugs', 'satisfactory'),
    ('RHnmuA3Y9Qg', 'satisfactory'),
    ('4oX_-JH0wVo', 'satisfactory'),
    ('4xyxT8o-JPhA', 'satisfactory')
]
steam_api = SteamAPI()
servico_s3 = ServicoS3()
servico_banco = OperacoesBancoDuckDb()
api_youtube = YoutubeAPI()

p1 = VerificarConexaoApiSteamCorrente(steam_api=steam_api)
p2 = ObterComentariosSteamCorrente(api_steam=steam_api, lista_jogos=lista_jogos)
p3 = GuardarDadosSteam3Corrente(
    servico_s3=servico_s3,
    servico_banco=servico_banco
)

p4 = ObterComentariosYoutubeCorrente(
    lista_jogos=lista_jogos_youtube,
    api_youtube=api_youtube
)

p5 = GuardarDadosYoutubeComentariosS3Corrente(
    servico_s3=servico_s3,
    servico_banco=servico_banco
)
p6 = ObterRespostaComentariosYoutubeCorrente(
    api_youtube=api_youtube
)
p7 = GuardarDadosYoutubeRespostaComentariosS3Corrente(
    servico_s3=servico_s3,
    servico_banco=servico_banco
)

p1.set_proxima_corrente(p2).set_proxima_corrente(p3).set_proxima_corrente(p4).set_proxima_corrente(
    p5).set_proxima_corrente(p6).set_proxima_corrente(p7)
p1.corrente(contexto=contexto)
