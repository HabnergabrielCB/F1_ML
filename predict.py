import pandas as pd
import joblib

def interactive_prediction():
    try:
        model = joblib.load('f1_model.pkl')
        encoder = joblib.load('f1_encoder.pkl')
        df_raw = pd.read_csv('f1_dataset_raw.csv')
    except FileNotFoundError:
        print("Erro: Rode o train.py primeiro para gerar os arquivos .pkl!")
        return

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
    
    # Filtra os pilotos da temporada mais recente registrada no CSV
    ultima_temp = df_raw['season'].max()
    df_recente = df_raw[df_raw['season'] == ultima_temp].sort_values(by='round')
    
    # Calcula a média de pontos recente de cada piloto
    df_recente['avg_points_last_3'] = df_recente.groupby('driver')['points'] \
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean()).fillna(0)
    
    grid_atual = df_recente.groupby('driver').last().reset_index()
    
    # Define a posição de largada simulada pela média de largada do piloto no ano
    grid_media = df_recente.groupby('driver')['grid_position'].mean().reset_index()
    grid_atual['grid_position'] = grid_atual['driver'].map(grid_media.set_index('driver')['grid_position']).round()
    
    # Define o clima médio histórico da pista escolhida
    clima_circuito = df_raw[df_raw['circuit'] == circuito_escolhido]
    grid_atual['circuit'] = circuito_escolhido
    grid_atual['air_temp'] = clima_circuito['air_temp'].mean()
    grid_atual['track_temp'] = clima_circuito['track_temp'].mean()
    grid_atual['rain'] = 0
    
    # Aplica o One-Hot Encoding e monta a tabela final para o modelo
    cat_cols = ['team', 'circuit']
    encoded_cats = encoder.transform(grid_atual[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
    
    num_features = ['grid_position', 'air_temp', 'track_temp', 'rain', 'avg_points_last_3']
    X_sim = pd.concat([grid_atual[num_features].reset_index(drop=True), encoded_df], axis=1)
    
    # Faz a predição
    grid_atual['podium_prob (%)'] = (model.predict_proba(X_sim)[:, 1] * 100).round(2)
    
    resultado = grid_atual[['driver', 'team', 'grid_position', 'podium_prob (%)']] \
        .sort_values(by='podium_prob (%)', ascending=False)
        
    print(f"\n=== Predição de Pódio: {circuito_escolhido} ===")
    print(resultado.to_string(index=False))

if __name__ == "__main__":
    interactive_prediction()