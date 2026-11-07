import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

def prepare_features(input_path='f1_dataset_raw.csv'):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['grid_position', 'final_position'])
    
    # 1. Ordenação cronológica obrigatória para séries temporais
    df = df.sort_values(by=['season', 'round'])
    
    # 2. Cálculo da Média Móvel (Rolling Average) das últimas 3 corridas
    # O shift(1) é vital: ele empurra os dados uma corrida para baixo. 
    # Sem isso, o modelo usaria os pontos da corrida de HOJE para prever a corrida de HOJE.
    df['avg_points_last_3'] = df.groupby('driver')['points'] \
                                .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    
    # Preenche NaN com 0 (ocorre na primeira corrida da base de cada piloto)
    df['avg_points_last_3'] = df['avg_points_last_3'].fillna(0)
    
    df['target_podium'] = (df['final_position'] <= 3).astype(int)
    
    cat_cols = ['team', 'circuit']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
    
    # 3. Adiciona a nova feature na lista de variáveis numéricas
    joblib.dump(encoder, 'f11_encoder.pkl')
    num_features = ['grid_position', 'avg_points_last_3']
    
    X = pd.concat([df[num_features].reset_index(drop=True), encoded_df], axis=1)
    y = df['target_podium'].reset_index(drop=True)
    
    return X, y, df[['season', 'round']].reset_index(drop=True)

if __name__ == "__main__":
    prepare_features()