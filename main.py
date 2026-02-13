from src.servicos.api_youtube.i_steam.steam_api import SteamAPI

if __name__ == '__main__':
    steam_api = SteamAPI()
    flag = steam_api.checar_conexao()
    print(flag)