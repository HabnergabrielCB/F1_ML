# 🏎️ F1 Podium Predictor — Machine Learning Pipeline

Um pipeline completo e modular de Machine Learning em Python para prever a probabilidade de pilotos da Fórmula 1 terminarem no pódio (Top 3) com base em dados históricos de largada, condições meteorológicas da pista e momento recente no campeonato.

## 📌 Arquitetura do Projeto

O projeto foi desenhado de forma modular para separar responsabilidades de engenharia de dados, processamento, treinamento e inferência:

* **`data_collector.py` (Extração e Cache):** Consome a API oficial de *timing* e resultados históricos da Fórmula 1 via `FastF1`, estruturando dados de grid, pontuação e telemetria climática (temperatura do ar, pista e chuva) com sistema de cache local para otimização de I/O.
* **`feature_engineering.py` (Pré-processamento e Séries Temporais):** Limpeza de dados, codificação de variáveis categóricas (`OneHotEncoder` para construtores e circuitos) e criação de janelas móveis (*rolling averages*) da pontuação das últimas 3 corridas de cada piloto, aplicando deslocamento temporal (`shift`) para evitar *Data Leakage*.
* **`train.py` (Modelagem e Validação Temporal):** Treinamento com **XGBoost Classifier** ajustado para classes desbalanceadas (`scale_pos_weight`, já que apenas 15% do grid sobe ao pódio). Utiliza corte estritamente cronológico entre treino e teste (validando em temporadas futuras reais em vez de *splits* aleatórios) e avalia a separabilidade via **ROC-AUC**.
* **`predict.py` (CLI Interativa de Inferência):** Simulador interativo via terminal onde o usuário seleciona o circuito desejado e o modelo carrega os artefatos serializados (`.pkl`) para projetar as probabilidades de pódio do grid atual.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Coleta de Dados:** FastF1
* **Manipulação e Engenharia de Features:** Pandas, Scikit-Learn
* **Algoritmo de Machine Learning:** XGBoost
* **Serialização de Artefatos:** Joblib

---

## 🚀 Como Executar Localmente

### 1. Clonar o repositório e criar o ambiente virtual
```bash
git clone [https://github.com/HabnergabrielCB/F1_ML.git](https://github.com/HabnergabrielCB/F1_ML.git)
cd F1_ML
python3 -m venv .venv
source .venv/bin/activate
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instalação das bibliotecas necessárias
pip install fastf1 pandas scikit-learn xgboost joblib

# Execução do Pipeline completo em ordem
python3 data_collector.py  # 1. Coletar e estruturar a base histórica (gera f1_dataset_raw.csv)
python3 train.py           # 2. Processar features e treinar o modelo (gera f1_encoder.pkl e f1_model.pkl)
python3 predict.py         # 3. Iniciar o simulador interativo de circuitos no terminal

# 📊 Conceitos de Engenharia de ML Aplicados neste projeto:
# - Prevenção de Data Leakage: Uso de Time-Series Split manual por temporada e lag features (shift(1)) no cálculo de médias móveis.
# - Desbalanceamento de Classes: Ajuste matemático de pesos na função de perda do XGBoost para lidar com a proporção 3:17 de pódios por corrida.
# - Consistência Treino-Serviço (Train-Serve Skew): Exportação conjunta do OneHotEncoder e do modelo treinado via joblib para garantir que a inferência em produção aplique a exata mesma matriz dimensional do treinamento.
