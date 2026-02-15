import time

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

inicio = time.perf_counter()

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
lista_jogos = [
    (1631270, "star_rupture"),
    (275850, "no_mans_sky"),
    (392160, "x4_foundations"),
    (526870, "satisfactory"),
    (1284190, "planet_crafter"),
    (4078590, "the_planet_crafter_toxicity"),
    (3142540, "the_planet_crafter_planet_humble"),
    (359320, "elite_dangerous"),
    (1336350, "elite_dangerous_odissey"),
    (227300, "euro_truck_simulator"),
    (2604420, "euro_truck_simulator_grecia"),
    (1209460, "euro_truck_simulator_iberia"),
    (558244, "euro_truck_simulator_italia"),
    (925580, "euro_truck_simulator_beyound_the_baltic_sea"),
    (1056760, "euro_truck_simulator_road_to_the_black_sea"),
    (531130, "euro_truck_simulator_vive_le_france"),
    (304212, "euro_truck_simulator_scandinaavia"),
    (227310, "euro_truck_simulator_going_east"),
    (2780810, "euro_truck_simulator_nordic_horizons"),
    (244850, "space_engineers"),
    (255710, "cities_skylines"),
    (264710, "subnautica"),
    (848450, "subnautica_bellow_zero"),
    (949230, "cities_skylines_dois"),
    (105600, "terraria"),
    (815370, "green_hell"),
    (396750, "everspace"),
    (1128920, "everspace_dois"),
    (281990, "stellaris"),
    (1363080, "manor_lords"),
    (108600, "project_zomboid"),
    (1149460, "icarus"),
    (361420, "astonomer"),
    (1172710, "dune_awakening"),
    (2570210, "eden_crafters"),
    (1203620, "enshrouded"),
    (1062090, "timberborn"),
    (1465470, "the_Crust"),
    (1783560, "the_last_caretaker"),
    (427520, "factorio"),
    (544550, "stationeers"),
    (2139460, "once_human"),
    (1466860, "age_of_empires_iv"),
    (1934680, "age_of_mythology_rethold"),
    (1244460, "jurassic_world_evolution_dois"),
    (2958130, "jurassic_world_evolution_tres"),
    (703080, "planet_zoo"),
    (3215050, "surviving_mars"),
    (1066780, "transport_fever_dois"),
    (1623730, "palword"),
    (1601580, "frostpunk_dois"),
    (1984270, "digimon_story_time_stranger"),
    (323190, "frostpunk"),
    (1125020, "frostpunk_the_rifts"),
    (1146960, "frostpunk_the_last_autumn"),
    (1147010, "frostpunk_on_the_edge"),
    (2791510, "frostpunk_dois_fractured_utopias"),
    (2277860, "frostpunk_dois_fractured_utopias"),
    (3417870, "stellaris_season_09_expansion_pass"),
    (3417860, "stellaris_stargazer_species_portrait"),
    (3417850, "stellaris_shadows_of_the_shroud"),
    (3417840, "stellaris_biogenesis"),
    (3283220, "stellaris_infernals_species_pack"),
    (2863190, "stellaris_season_08_expansion_pass"),
    (2863180, "stellaris_rick_the_cube_species_portrait"),
    (2863080, "stellaris_grand_archive_story_pack"),
    (2863060, "stellaris_cosmic_storms"),
    (2840100, "stellaris_the_machine_age"),
    (2729490, "stellaris_expansion_subscription"),
    (2534090, "stellaris_astral_planes"),
    (2380030, "stellaris_galactic_paragons"),
    (2277860, "stellaris_first_contact_story_pack"),
    (2115770, "stellaris_toxoids_species_pack"),
    (1889490, "stellaris_overlord"),
    (1749080, "stellaris_aquatics_species_pack"),
    (1522090, "stellaris_nemesis"),
    (1341520, "stellaris_necroids_species_pack"),
    (1140001, "stellaris_federations"),
    (1140000, "stellaris_lithoids_species_pack"),
    (1045980, "stellaris_ancient_relics_story_pack"),
    (944290, "stellaris_megacorp"),
    (844810, "stellaris_distant_stars_story_pack"),
    (756010, "stellaris_humanoids_species_pack"),
    (716670, "stellaris_apocalypse"),
    (642750, "stellaris_synthetic_dawn_story_pack"),
    (633310, "stellaris_anniversary_portraits"),
    (616191, "stellaris_galaxy_edition_upgrade_pack"),
    (616190, "stellaris_nova_edition_upgrade_pack"),
    (554350, "stellaris_horizon_signal"),
    (553280, "stellaris_utopia"),
    (518910, "stellaris_leviathans_story_pack"),
    (498870, "stellaris_plantoids_species_pack"),
    (497660, "stellaris_infinite_frontiers_ebook"),
    (462720, "stellaris_creatures_of_the_void"),
    (461461, "stellaris_galaxy_preorder_termination_100388"),
    (461073, "stellaris_nova_preorder_termination_99329"),
    (461071, "stellaris_preorder_99330"),
    (447750, "stellaris_preview_depot"),
    (447688, "steamdb_unknown_app_447688_corrupt_depot"),
    (447687, "stellaris_ringtones"),
    (447686, "stellaris_novel_by_steven_savile"),
    (447685, "stellaris_signed_highres_wallpaper"),
    (447684, "stellaris_digital_ost"),
    (447683, "stellaris_arachnoid_portrait_pack"),
    (447682, "stellaris_digital_artbook"),
    (447681, "stellaris_signup_campaign_bonus"),
    (447680, "stellaris_symbols_of_domination"),
    (3778100, "x4_envoy_pack"),
    (3419300, "x4_hyperion_pack"),
    (2700340, "x4_timelines"),
    (2375830, "x4_community_of_planets_collectors_edition_bonus_content"),
    (1990040, "x4_kingdom_end"),
    (1701060, "x4_tides_of_avarice"),
    (1288460, "x4_cradle_of_humanity"),
    (1133000, "x4_split_vendetta"),
    (986330, "x4_foundations_preorder_bonus"),
    (942190, "x4_foundations_collectors_edition_content")

]
lista_jogos_youtube = [
    # ('WLilIKOJYi0', 'star_rupture'),
    # ('D1PDHTGNswI', 'star_rupture'),
    # ('9Ib27hgkG3s', 'star_rupture'),
    # ('xpo9nSt7Das', 'star_rupture'),
    # ('ri0cpSX2ero', 'star_rupture'),
    # ('0YfO99f2QSU', 'star_rupture'),
    # ('vZLYbXaaXLA', 'star_rupture'),
    # ('jxx8ue0SwnA', 'star_rupture'),
    # ('DvbJDTxb7is', 'star_rupture'),
    # ('w5yNWwJn7V8', 'star_rupture'),
    # ('xdZoDbeJ-oM', 'star_rupture'),
    # ('OvHk_itOugs', 'satisfactory'),
    # ('RHnmuA3Y9Qg', 'satisfactory'),
    # ('4oX_-JH0wVo', 'satisfactory'),
    # ('xyxT8o-JPhA', 'satisfactory'),
    # ('yB97xvhGf3s', 'satisfactory'),
    # ('lF0hBETvOuQ', 'satisfactory'),
    # ('jEFz4PVixrQ', 'satisfactory'),
    # ('7Ulg3PRYo80', 'satisfactory'),
    # ('HAYetmTD_og', 'no_mans_sky'),
    # ('ktsrYBZcXSE', 'no_mans_sky'),
    # ('3eox_HfBOck', 'no_mans_sky'),
    # ('6XoSPVADxL8', 'no_mans_sky'),
    # ('xjrV9d-ZVzg', 'no_mans_sky'),
    # ('04y9zgwsqU4', 'satisfactory'),
    ('YI6delmSrc0', 'subnautica'),
    ('yWRaxYP-wns', 'subnautica'),
    ('ubVYWktpNL0', 'subnautica'),
    ('lYXT1si2c7U', 'subnautica'),
    ('zp5fBG7T5tk', 'subnautica'),
    ('zas4Oxz3Uvs', 'stellaris'),
    ('HZZzO3pG8lE', 'factorio'),
    ('OqU9tESMELk', 'factorio'),
    ('yQiLF9D5TqU', 'factorio'),
    ('Czlu0lBHJqQ', 'factorio'),
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

# p1.set_proxima_corrente(p2).set_proxima_corrente(p3)
p1.corrente(contexto=contexto)
fim = time.perf_counter()
tempo_total = fim - inicio

print(f"\n⏱ Tempo total de execução: {tempo_total:.4f} segundos")
