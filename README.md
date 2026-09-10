# fraud-detection-project

**Pipeline completo de machine learning para detecção de fraude em transações de cartão de crédito — do download do dataset ao modelo serializado em produção.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-4DABF7?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=flat-square)
![joblib](https://img.shields.io/badge/persist%C3%AAncia-joblib-094E5C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)

Projeto de machine learning end-to-end que treina e compara três classificadores para detectar fraude em transações de cartão de crédito, usando o dataset público Credit Card Fraud Detection (Kaggle/mlg-ulb). O foco técnico é o que separa este problema de uma classificação comum: **class imbalance extrema** (0,173% de fraudes), seleção de métrica orientada ao negócio (Recall como métrica principal, F1 como secundária) e um pipeline reprodutível em um único comando — download do CSV (~67 MB), EDA, pré-processamento, balanceamento, busca de hiperparâmetros com RandomizedSearchCV e serialização do modelo vencedor em `.joblib`. Stack: Python 3.9+, pandas 2.0+, NumPy, scikit-learn 1.3+, Matplotlib, Seaborn e joblib. Ideal para demonstrar domínio de pipeline ML completo em processos seletivos de dados/IA.

## 🇧🇷 Português

### Visão geral

O `main.py` orquestra o pipeline inteiro, do dado bruto ao artefato treinado:

1. **Download automático do dataset** — busca `creditcard.csv` (~67 MB) de `storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` caso ainda não exista localmente.
2. **EDA** — análise exploratória com gráficos de distribuição de classes, montante e tempo (saída em `plots/`).
3. **Pré-processamento** — remoção de duplicatas e escalonamento de features.
4. **Balanceamento de classes** — `class_weight='balanced'` como estratégia principal; SMOTE opcional via `imbalanced-learn` para comparação.
5. **Treino e comparação de 3 modelos** — LogisticRegression, RandomForest e ExtraTrees, todos com `RandomizedSearchCV` para busca de hiperparâmetros.
6. **Seleção por pontuação ponderada** — Recall como métrica principal, F1 como secundária; o vencedor é escolhido por pontuação ponderada, não por accuracy isolada.
7. **Persistência** — modelo campeão salvo como `.joblib`; gráficos de avaliação em `plots/`.

O `inspecionar_modelo.py` carrega o modelo salvo e visualiza os resultados persistidos sem retreinar.

### Dataset

Credit Card Fraud Detection (Kaggle/mlg-ulb) — transações de portadores de cartão europeus, coletadas em dois dias de setembro 2013.

| Campo | Valor |
|---|---|
| Transações totais | 284.807 |
| Fraudadoras | 492 (0,173%) |
| Legítimas | 284.315 (99,827%) |
| Features | `Time`, `V1`–`V28` (componentes PCA, preservação de privacidade), `Amount` |
| Target | `Class` (1 = fraude, 0 = legítima) |

O desbalanceamento de ~1:578 é o desafio central do projeto: um classificador trivial que prevê "legítima" para tudo acerta 99,8% das vezes e não detecta fraude nenhuma. Por isso accuracy é uma métrica enganosa aqui, e o pipeline prioriza Recall (taxa de detecção) e F1.

### Resultados reais

Comparação obtida na execução do pipeline (RandomizedSearchCV, validação com estratificação):

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | Tempo |
|---|---|---|---|---|---|---|
| **RandomForest (vencedor)** | **0,9994** | **0,8046** | **0,7368** | **0,8046** | **0,9445** | ~35 s |
| LogisticRegression | 0,9752 | 0,0562 | 0,8737 | 0,1057 | 0,9657 | ~4 s |
| ExtraTrees | 0,9987 | 0,6667 | 0,8000 | 0,6667 | 0,9621 | ~3 s |

Hiperparâmetros do RandomForest campeão: `class_weight='balanced'`, `max_depth=10`, `min_samples_split=2`, `n_estimators=50`.

Leitura dos números: a regressão logística tem o maior Recall bruto (0,8737) e o maior ROC AUC (0,9657), mas dispara tantos falsos positivos que o F1 desaba para 0,1057 — em operação, isso significa revisar manualmente quase toda a base. O RandomForest equilibra Precision 0,8046 com Recall 0,7368, capturando ~74% das fraudes com pouquíssimos alertas incorretos, e vence pela pontuação ponderada.

### Como reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/KelvinOliveiraCode/fraud-detection-project.git
cd fraud-detection-project

# 2. (Opcional) Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o pipeline completo
python main.py

# 5. (Opcional) Inspecione o modelo salvo sem retreinar
python inspecionar_modelo.py
```

O dataset (~67 MB) é baixado automaticamente no primeiro run — não é preciso baixar nada manualmente. Execução completa leva de **3 a 8 minutos** dependendo do hardware, já incluindo o RandomizedSearchCV dos três modelos.

### Decisões de design

- **Recall como métrica principal**: em detecção de fraude, o custo de uma fraude não detectada (perda financeira direta) supera o custo de um falso positivo (revisão manual). A função de seleção reflete isso com pontuação ponderada em Recall e F1.
- **`class_weight='balanced'` como padrão**: penalizar erros na classe minoritária diretamente na função de perda evita distorcer a distribuição real dos dados com oversampling sintético. SMOTE fica como camada opcional via `imbalanced-learn` para comparação experimental.
- **Três famílias de modelo**: linear (LogisticRegression), bagging de árvores profundas (RandomForest) e extra trees (ExtraTrees) — famílias com viés/variância distintos dão uma comparação honesta, em vez de três variações do mesmo algoritmo.
- **RandomizedSearchCV em vez de GridSearchCV**: com 284 mil amostras, busca exaustiva é inviável em hardware comum; amostragem aleatória do espaço de hiperparâmetros cobre o grid de forma eficiente.
- **Remoção de duplicatas antes do split**: transações duplicadas vazariam informação entre treino e teste e inflariam as métricas.
- **joblib para persistência**: serialização eficiente de pipelines scikit-learn com arrays NumPy; o `.joblib` recarrega o modelo pronto para inferência.

### Estrutura do repositório

```
fraud-detection-project/
├── main.py               # Pipeline completo: download → EDA → treino → seleção → save
├── inspecionar_modelo.py # Carrega o modelo salvo e visualiza os resultados
├── requirements.txt      # Dependências (pandas, scikit-learn, imbalanced-learn etc.)
├── LICENSE               # MIT
└── plots/                # Gráficos gerados na execução
```

---

## 🇺🇸 English

### Overview

`main.py` runs the entire pipeline, raw data to trained artifact:

1. **Automatic dataset download** — fetches `creditcard.csv` (~67 MB) from `storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` if missing locally.
2. **EDA** — class distribution, amount, and time plots (written to `plots/`).
3. **Preprocessing** — duplicate removal and feature scaling.
4. **Class imbalance handling** — `class_weight='balanced'` as the primary strategy; optional SMOTE via `imbalanced-learn` for comparison.
5. **Three-model training and comparison** — LogisticRegression, RandomForest, and ExtraTrees, all tuned with `RandomizedSearchCV`.
6. **Weighted-score selection** — Recall as the primary metric, F1 secondary; the winner is picked by a weighted score, never by accuracy alone.
7. **Persistence** — best model saved as `.joblib`; evaluation plots in `plots/`.

`inspecionar_modelo.py` reloads the saved model and visualizes stored results without retraining.

### Dataset

Credit Card Fraud Detection (Kaggle/mlg-ulb) — European cardholder transactions over two days, September 2013.

| Field | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.173%) |
| Legitimate | 284,315 (99.827%) |
| Features | `Time`, `V1`–`V28` (PCA components for privacy), `Amount` |
| Target | `Class` (1 = fraud, 0 = legitimate) |

The ~1:577 imbalance is the core challenge: a trivial classifier predicting "legitimate" for everything scores 99.8% accuracy and catches zero fraud. Accuracy is meaningless here — the pipeline optimizes Recall (detection rate) and F1.

### Actual results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Time |
|---|---|---|---|---|---|---|
| **RandomForest (winner)** | **0.9994** | **0.8046** | **0.7368** | **0.8046** | **0.9445** | ~35 s |
| LogisticRegression | 0.9752 | 0.0562 | 0.8737 | 0.1057 | 0.9657 | ~4 s |
| ExtraTrees | 0.9987 | 0.6667 | 0.8000 | 0.6667 | 0.9621 | ~3 s |

Winning RandomForest hyperparameters: `class_weight='balanced'`, `max_depth=10`, `min_samples_split=2`, `n_estimators=50`.

How to read it: logistic regression has the highest raw Recall (0.8737) and ROC AUC (0.9657), but floods the queue with false positives — F1 collapses to 0.1057, which in production means manually reviewing almost the entire base. RandomForest balances Precision 0.8046 with Recall 0.7368: ~74% of fraud caught with very few wrong alerts, winning on the weighted score.

### Reproduce

```bash
git clone https://github.com/KelvinOliveiraCode/fraud-detection-project.git
cd fraud-detection-project

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
python main.py                   # full pipeline
python inspecionar_modelo.py     # inspect saved model, no retraining
```

The dataset (~67 MB) downloads automatically on the first run. Full execution takes **3–8 minutes** depending on hardware, RandomizedSearchCV included.

### Design decisions

- **Recall as the primary metric**: in fraud detection, the cost of a missed fraud (direct financial loss) outweighs a false positive (manual review). The selection function encodes this via a weighted score over Recall and F1.
- **`class_weight='balanced'` by default**: penalizing minority-class errors in the loss function keeps the real data distribution intact, unlike synthetic oversampling. SMOTE stays optional via `imbalanced-learn` for experiments.
- **Three model families**: linear (LogisticRegression), deep-tree bagging (RandomForest), and extremely randomized trees (ExtraTrees) — families with distinct bias/variance profiles make the comparison honest.
- **RandomizedSearchCV over GridSearchCV**: with 284k samples, exhaustive search is impractical on commodity hardware; random sampling covers the space efficiently.
- **Deduplication before the split**: duplicate transactions leaking between train and test would inflate metrics.
- **joblib persistence**: efficient scikit-learn pipeline serialization with NumPy arrays; the `.joblib` reloads inference-ready.

---

## Autor

**Kelvin Oliveira**

- GitHub: [KelvinOliveiraCode](https://github.com/KelvinOliveiraCode)
- LinkedIn: [kelvin-oliveira-code](https://www.linkedin.com/in/kelvin-oliveira-code/)

## Licença

Distribuído sob a licença [MIT](LICENSE).
