#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:20:39 2026

@author: rodrigo
"""
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb
import pandas as pd

obddb = OperacoesBancoDuckDb()

# Tratamento Comentário

caminho_consulta = f's3://extracao/youtube/bronze/comentarios_youtube/*/*/comentarios.json'


dataframe_comentarios = obddb.consultar_dados('1=1', caminho_consulta)
df_snippet_comentarios = pd.json_normalize(
    dataframe_comentarios['snippet'],
    sep='_'
)

df_comentarios_final = df_snippet_comentarios[['channelId','videoId','topLevelComment_id' , 'topLevelComment_snippet_textDisplay' ]]

df_comentarios_final.rename(
    columns={
        "channelId": 'id_canal',
        "videoId": "id_video",
        "topLevelComment_id": "id_comentario",
        "topLevelComment_snippet_textDisplay": "texto_comentario"
        
        
        },
    inplace=True
   )


# Tratamento_resposta_comentarios
caminho_consulta_resposta_comentarios  = f's3://extracao/youtube/bronze/resposta_comentarios_youtube/*/*/*/resposta_comentarios.json'
dataframe_reposta_comentarios = obddb.consultar_dados('1=1', caminho_consulta_resposta_comentarios)
df_snippet_resposta_comentarios = pd.json_normalize(
    data=dataframe_reposta_comentarios[['snippet', 'id']].dropna(subset=['snippet']).to_dict(orient='records'),
    sep='_'
)

df_snippet_resposta_comentarios['id_comentario'] = df_snippet_resposta_comentarios['id'].str.split('.').str[1]


df_snippet_resposta_comentarios = df_snippet_resposta_comentarios[['snippet_channelId', 'snippet_parentId', 'id_comentario',  'snippet_textDisplay']]

df_resposta_comentarios_final = pd.merge(
    df_comentarios_final[['id_video', 'id_comentario']], 
    df_snippet_resposta_comentarios, 
    left_on='id_comentario',
    right_on='snippet_parentId',
    how='inner'
    ) 

df_resposta_comentarios_final = df_resposta_comentarios_final[['snippet_channelId', 'id_video', 'id_comentario_x', 'snippet_textDisplay']]

df_resposta_comentarios_final.rename(columns={
    'snippet_channelId': 'id_canal',
    'videoId': 'id_video',
    'snippet_textDisplay': 'texto_comentario',
    'id_comentario_x': 'id_comentario'
    }, inplace=True)

#união dataframe

df_comentarios_tratado_final = pd.concat([df_resposta_comentarios_final, df_comentarios_final])
df_comentarios_tratado_final.to_csv('df_comentarios_tratado_final.csv', sep='|')