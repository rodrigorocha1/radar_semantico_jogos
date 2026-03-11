# Mapa Semântico de Tendências em Games

Monitorar e analisar comentários de vídeos de jogos no YouTube e reviews na Steam, transformando dados não estruturados em insights sobre reputação, sentimentos e tendências, com visualização via mapa auto-organizável (SOM), para identificar padrões de opinião, sentimentos e tendências em relação a diferentes jogos e gêneros. O projeto inclui rastreamento de modelos via MLflow para versionamento de análises e resultados.


# Arquitetura da Solução

### Coleta de Dados

Para inicio do projeto, foi organizado a coleta dos dados dos textos do reviews dos jogos (Tabela abaixo), usando o endpoint: https://store.steampowered.com/appreviews/10?json=1&filter=recent&cursor=*&review_type=all&purchase_type=all&num_per_page=100&day_range=365 r  a api de comentários de vídeo do youtube:
youtube/v3/commentThreads
youtube/v3/comments


usando o padrão de projetos cadeia de responsabilidade e o guardando do json em um bucket, com Minio.

Tabela com os jogos que serão analisados:

| AppID | Nome Técnico (Slug) | Franquia / Categoria |
|------|------|------|
| 1631270 | star_rupture | Star Rupture |
| 275850 | no_mans_sky | No Man’s Sky |
| 392160 | x4_foundations | X4 |
| 526870 | satisfactory | Satisfactory |
| 1284190 | planet_crafter | The Planet Crafter |
| 4078590 | the_planet_crafter_toxicity | The Planet Crafter (DLC) |
| 3142540 | the_planet_crafter_planet_humble | The Planet Crafter (DLC) |
| 359320 | elite_dangerous | Elite Dangerous |
| 1336350 | elite_dangerous_odissey | Elite Dangerous (DLC) |
| 227300 | euro_truck_simulator | Euro Truck Simulator 2 |
| 2604420 | euro_truck_simulator_grecia | ETS2 (DLC) |
| 1209460 | euro_truck_simulator_iberia | ETS2 (DLC) |
| 558244 | euro_truck_simulator_italia | ETS2 (DLC) |
| 925580 | euro_truck_simulator_beyound_the_baltic_sea | ETS2 (DLC) |
| 1056760 | euro_truck_simulator_road_to_the_black_sea | ETS2 (DLC) |
| 531130 | euro_truck_simulator_vive_le_france | ETS2 (DLC) |
| 304212 | euro_truck_simulator_scandinaavia | ETS2 (DLC) |
| 227310 | euro_truck_simulator_going_east | ETS2 (DLC) |
| 2780810 | euro_truck_simulator_nordic_horizons | ETS2 (DLC) |
| 244850 | space_engineers | Space Engineers |
| 255710 | cities_skylines | Cities Skylines |
| 264710 | subnautica | Subnautica |
| 848450 | subnautica_bellow_zero | Subnautica (Below Zero) |
| 949230 | cities_skylines_dois | Cities Skylines II |
| 105600 | terraria | Terraria |
| 815370 | green_hell | Green Hell |
| 396750 | everspace | Everspace |
| 1128920 | everspace_dois | Everspace 2 |
| 281990 | stellaris | Stellaris |
| 1363080 | manor_lords | Manor Lords |
| 108600 | project_zomboid | Project Zomboid |
| 1149460 | icarus | Icarus |
| 361420 | astonomer | Astroneer |
| 1172710 | dune_awakening | Dune Awakening |
| 2570210 | eden_crafters | Eden Crafters |
| 1203620 | enshrouded | Enshrouded |
| 1062090 | timberborn | Timberborn |
| 1465470 | the_Crust | The Crust |
| 1783560 | the_last_caretaker | The Last Caretaker |
| 427520 | factorio | Factorio |
| 544550 | stationeers | Stationeers |
| 2139460 | once_human | Once Human |
| 1466860 | age_of_empires_iv | Age of Empires IV |
| 1934680 | age_of_mythology_rethold | Age of Mythology Retold |
| 1244460 | jurassic_world_evolution_dois | Jurassic World Evolution 2 |
| 2958130 | jurassic_world_evolution_tres | Jurassic World Evolution 3 |
| 703080 | planet_zoo | Planet Zoo |
| 3215050 | surviving_mars | Surviving Mars |
| 1066780 | transport_fever_dois | Transport Fever 2 |
| 1623730 | palword | Palworld |
| 1601580 | frostpunk_dois | Frostpunk 2 |
| 1984270 | digimon_story_time_stranger | Digimon Story |
| 323190 | frostpunk | Frostpunk |
| 1125020 | frostpunk_the_rifts | Frostpunk (DLC) |
| 1146960 | frostpunk_the_last_autumn | Frostpunk (DLC) |
| 1147010 | frostpunk_on_the_edge | Frostpunk (DLC) |
| 2791510 | frostpunk_dois_fractured_utopias | Frostpunk 2 (DLC) |
| 3417870 → 447680 | stellaris_* | Stellaris (DLCs / Packs diversos) |
| 3778100 → 942190 | x4_* | X4 (DLCs / Packs diversos) |
| 427100 | fernbus_simulator | Fernbus Simulator |
| 492720 | tropico_seis | Tropico 6 |


## Interpretação do resultado 



![qe_te](http://raw.githubusercontent.com/rodrigorocha1/radar_semantico_jogos/refs/heads/master/img/qe_te.png)


O gráfico mostra o comportamento das métricas, erro de quantização e erro topográfico. O erro de quantização mede a distância média entre os dados de entrada e os neurônios vencedores.
No gráfico, o valor começa perto de 0.47 e apresenta uma queda constante ao logo de 600 épocas estabilizando perto de 0.34, mostrando que a rede está aprendendo  e os pesos dos neurônios estão se ajustando para representar o espaço vetorial dos embedding originais;
O erro topográfico avalia a preservação da topologia da rede. O resultado fica aproximadamente entre 2% e 5% dos dados ativam bmus que não são adjacentes. Com esses resultados, o mapa conseguiu preservar as relações de vizinhança dos dados,, ou seja, comentários similares, estão em neurônios próximos no mapa.


## Macrotemas

Para a interpretação do mapa auto organizável, eu organizei em seguintes macrotemas:

---

### Macrotema 0: Comentários curtos, expressões rápidas e idiomas estrangeiros

Este agrupamento reúne textos curtos, diretos e emocionais. É marcado por girias da internet e o uso de caixa alta.

**Exemplos:**

- "Very good, even better after the last update."
- "cometi suicidio 10/10"
- "Nota mil, coisa linda"

---

### Macrotema 1: Reviews Profundos, Análises Técnicas e Relatos Detalhados

Concentra as avaliações e criticas sobre jogos, debate spbre mecanica de jogos (gerenciamento, sobreviência e simulação), discursão sobre pontos positivos e negátivos e discursos sobre dlcs e ótimização.

**Ex:**

- Gostei muito, e de jogar com amigos ainda mais.

- Um DLC que demonstra a Península Ibérica! Portugal pode estar incompleto, mas o esforço está a lá, também sabendo que a escala do mapa é um bocado rigorosa, fica um bocado difícil.  
Duas cidades serão adicionadas gratuitamente, e acho este ato com muito respeito!  
Recomendava adicionar a cidade de Lagos, porque tem mais à frente uma rotunda cruzando com a A22 com a N121, que liga para Sines.  
Do resto, In-cri-vél!

- Já tinha ficado impressionado com o salto de qualidade quando comprei a expansão Scandinavia mas esta... ultrapassa tudo o que já vi neste jogo !  
São as empresas e cargas novas, é a vegetação mais densa, mais verde, são as áreas de serviço que agora sim são largas e dignas desse nome, é a vasta rede de estradas e auto-estradas (13.500km), as 5 novas conquistas para horas e horas de jogo e o cuidado que tiveram em colocar certos pontos míticos da França no jogo (pena apenas não terem colocado em Paris a Torre Eiffel).  

E que lindas que são as aldeias rústicas com as suas casinhas de pedra !  

Com isto tudo, é caso para dizer: Vive la France !

---

### Macrotema 2: Interações com os criadores (Youtube) e Avaliações moderadas

Aqui, os comentários são majoritariamente em português, focados em interações sociais e opiniões diretas, mas sem grandes detalhamentos técnicos. Também reside a grande maioria dos comentários claramente originados do YouTube (ex: conversando com o dono do canal, parabenizando pelo vídeo, discutindo a série) ou avaliações comuns e moderadas ("bom jogo", "recomendo muito").

- Tenho esse eden crafters ainda não joguei ele tenho a bastante tempo parece ser bom
- Espera-vos muito divertimento
- Boa tarde Roma

## U-matrix

![u_matrix](https://raw.githubusercontent.com/rodrigorocha1/radar_semantico_jogos/refs/heads/master/img/u_matrix.png)

## Regiões de Comportamento na U-Matrix

Com base na u-matrix, foram separados 3 regiões de comportamento e comentários dos usuários.

### Região 1  
**(Região superior do mapa, linhas 0-10 e colunas 0-9): Comentários gerais sobre jogos**

É composto por discursão do jogo , comentários são longos e mistrura de opiniões, experiências de gameplay, humor e avaliações detalhadas.

---

### Região 2  
**(Região central esquerda linhas 14 -20 e colunas 1-3): Satistfação e elogiões rápidos**

É composto por feedback positivo simples, contém comentários curtos, foco em aprovação geral e pouca argumentação.

---

### Região 3  
**(Região inferior direita da u-matrix Linhas 21 – 25 e colunas 6 - 9):**

Contém reações emocionais fortes, opiniões extremas, linguagem informal ou impulsiva e mistura de entusiasmo re rejeição

## Hit map 
![hit_map](https://raw.githubusercontent.com/rodrigorocha1/radar_semantico_jogos/refs/heads/master/img/hit_map.png)

## Detalhamento das Regiões do Mapa

### Região superior do mapa  
**Faixa:** Linha 0 e Colunas 0 – 10  

Detecção de elementos de gameplay, como evento de expedições e experiências pessoais no jogo, indicando usuários engajados e interesse no conteúdo.

---

### Região inferior do mapa  
**Faixa:** Linha 25 e Colunas 0 – 10  

Detecção de forte engajamento do público, além de detecção de momentos marcantes de uma luta e elementos de sobrevivência.)


# Demonstração do projeto
[![Assistir ao vídeo de demonstração do projeto](https://img.shields.io/badge/🎬%20Assistir%20ao%20vídeo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/BfgtWVOKMDo)