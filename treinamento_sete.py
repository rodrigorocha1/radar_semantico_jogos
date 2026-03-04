from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Seus dados de comentários por neurônio
comentarios_map = {
    "(0, 0)": "Pare de procurar analises ruins, esse jogo não tem defeitos, apenas compre imediatamente.",
    "(0, 1)": "gostaria que em atualizações posteriores, o modo de como o duke fishron é invocado mudasse...",
    "(0, 2)": "Terraria é muito mais do que um simples jogo 2D. É uma experiência profunda, versátil e altamente viciante...",
    "(0, 3)": "Jogo Maravilhoso eu adorei a Gameplay dele é incrível além de que o suspense e o mistério...",
    "(0, 4)": "Jogo Ótimo, Comprei um pc, baixei a steam, comprei o jogo a preço de pão...",
    "(1, 0)": "Massa demais!",
    "(1, 2)": "Pra quem gosta de pescar a missão do pescador é perfeita, mas sinceramente...",
    "(1, 3)": "melhor jogo da historia",
    "(1, 4)": "jogo véi peba",
    "(2, 0)": "rumo as 1000 horas de jogatina",
    "(2, 1)": "legal",
    "(2, 2)": "tipo minecraft",
    "(2, 4)": "bom jogo!!!",
    "(3, 0)": "incrivel jogari mais",
    "(3, 1)": "Bacana.",
    "(3, 2)": "MARAVILHOSO",
    "(3, 3)": "Terra.",
    "(3, 4)": "muito bom o jogo super recomendo",
    "(4, 0)": "muitobom",
    "(4, 1)": "Jogo otimo, Maravilhoso, é um Minecraft 1 google melhor",
    "(4, 2)": "é terraria né, n tem nem oq falar",
    "(4, 3)": "Por mais que tenha só 1hora de gameplay, eu já platinei esse jogo no celular...",
    "(4, 4)": "muito bom tem muita coisa para fazer uns bosses um pouco dificils mas muito bom",
}

# Descobrir tamanho do grid
linhas = max(int(k.split(',')[0][1:]) for k in comentarios_map.keys()) + 1
colunas = max(int(k.split(',')[1][:-1]) for k in comentarios_map.keys()) + 1

fig, axes = plt.subplots(linhas, colunas, figsize=(colunas*6, linhas*6))

for i in range(linhas):
    for j in range(colunas):
        ax = axes[i, j]
        key = f"({i}, {j})"
        ax.axis("off")
        if key in comentarios_map:
            wc = WordCloud(
                width=400,
                height=400,
                background_color="white",
                colormap="plasma",
                max_font_size=80,    # Aumenta o tamanho máximo da fonte
                min_font_size=10,
                font_step=2
            ).generate(comentarios_map[key])
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"Posição -> {key}", fontsize=16, fontweight='bold')  # Título maior

plt.tight_layout()
plt.savefig('exemplo.png')
plt.close()