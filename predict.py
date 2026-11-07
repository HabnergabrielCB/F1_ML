import os
import pandas as pd
import joblib

def interactive_prediction():
    # Verifica se os arquivos necessários existem na pasta
    for arq in ['f1_dataset_raw.csv', 'f1_encoder.pkl', 'f1_model.pkl']:
        if not os.path.exists(arq):
            print(f"ERRO: O arquivo '{arq}' não está na pasta!")
            return

    # Carrega modelo, encoder e dados brutos
    model = joblib.load('f1_model.pkl')
    encoder = joblib.load('f1_encoder.pkl')
    df_raw = pd.read_csv('f1_dataset_raw.csv')

    # Menu interativo de seleção de circuito
    circuitos = sorted(df_raw['circuit'].unique())
    print("\n=== Escolha o Circuito para Simulação ===")
    for idx, circuito in enumerate(circuitos, 1):
        print(f"[{idx}] {circuito}")
        
    try:
        escolha = int(input("\nDigite o número do circuito: ")) - 1
        if escolha < 0 or escolha >= len(circuitos):
            print("Número inválido.")
            return
        circuito_escolhido = circuitos[escolha]
    except ValueError:
        print("Entrada inválida. Digite apenas números.")
        return
    
    # Filtra os dados da temporada mais recente
    ultima_temp = df_raw['season'].max()
    df_recente = df_raw[df_raw['season'] == ultima_temp].sort_values(by='round')
    
    # Isola estritamente os 20 pilotos que participaram da última corrida real
    ultima_corrida = df_recente['round'].max()
    pilotos_atuais = df_recente[df_recente['round'] == ultima_corrida]['driver'].unique()
    df_recente = df_recente[df_recente['driver'].isin(pilotos_atuais)]
    
    # Calcula a média móvel de pontos (últimas 3 corridas)
    df_recente['avg_points_last_3'] = df_recente.groupby('driver')['points'] \
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean()).fillna(0)
    
    grid_atual = df_recente.groupby('driver').last().reset_index()
    
    # Calcula a média de largada e aplica ranqueamento de 1 a 20 (sem empates)
    grid_media = df_recente.groupby('driver')['grid_position'].mean().reset_index()
    grid_media['grid_position_simulada'] = grid_media['grid_position'].rank(method='first').astype(int)
    
    # Atribui o novo grid e o circuito escolhido
    grid_atual['grid_position'] = grid_atual['driver'].map(grid_media.set_index('driver')['grid_position_simulada'])
    grid_atual['circuit'] = circuito_escolhido
    
    # Processa as variáveis categóricas
    cat_cols = ['team', 'circuit']
    encoded_cats = encoder.transform(grid_atual[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
    
    # Concatena a matriz final (usando as variáveis numéricas sem clima)
    num_features = ['grid_position', 'avg_points_last_3']
    X_sim = pd.concat([grid_atual[num_features].reset_index(drop=True), encoded_df], axis=1)
    
    # Gera as probabilidades do XGBoost
    grid_atual['podium_prob (%)'] = (model.predict_proba(X_sim)[:, 1] * 100).round(2)
    
    resultado = grid_atual[['driver', 'team', 'grid_position', 'podium_prob (%)']] \
        .sort_values(by='podium_prob (%)', ascending=False)
        
    print(f"\n=== Predição de Pódio: {circuito_escolhido} ===")
    print(resultado.to_string(index=False))

if __name__ == "__main__":
    interactive_prediction()