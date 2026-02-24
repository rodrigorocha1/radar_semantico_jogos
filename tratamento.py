import pandas as pd

from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb

a = 1


obddb = OperacoesBancoDuckDb()

# Tratamento Comentário

caminho_consulta = f's3://extracao/comentarios/prata/comentarios_limpos_2026_02_23_20_32_35.csv'


dataframe_comentarios = obddb.consultar_dados('1=1', caminho_consulta)


print(dataframe_comentarios.shape)
