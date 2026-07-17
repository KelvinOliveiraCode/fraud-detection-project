# Detector de Fraudes Bancarias - Credit Card Fraud Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-3.7%2B-green?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/pandas-2.0%2B-purple?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>Projeto de Machine Learning para deteccao de fraudes em transacoes de cartao de credito</b><br>
  Desenvolvido como projeto de conclusao de curso na plataforma <strong>DIO (Digital Innovation One)</strong>
</p>

---

## Sumario

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalacao](#instalacao)
- [Como Usar](#como-usar)
- [Passo a Passo do Desenvolvimento](#passo-a-passo-do-desenvolvimento)
- [Resultados](#resultados)
- [Autor](#autor)
- [Licenca](#licenca)

---

## Sobre o Projeto

Este projeto implementa um **sistema completo de deteccao de fraudes bancarias** utilizando tecnicas de Machine Learning. O objetivo e identificar transacoes fraudulentas em cartoes de credito com alta precisao, maximizando o **Recall** (taxa de deteccao de fraudes) e mantendo um bom equilibrio entre as metricas.

### Objetivos do Projeto

- [x] Construir um pipeline completo de Machine Learning
- [x] Tratar dados altamente desbalanceados (99.8% legitimas vs 0.17% fraudes)
- [x] Comparar multiplos modelos e selecionar o melhor
- [x] Otimizar hiperparametros automaticamente
- [x] Gerar visualizacoes e relatorios completos
- [x] Salvar o modelo para uso em producao

---

## Dataset

O dataset utilizado e o **Credit Card Fraud Detection** da European Cardholders, disponibilizado pela [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e pela Google.

| Caracteristica | Valor |
|---|---|
| **Total de transacoes** | 284.807 |
| **Fraudes** | 492 (0.173%) |
| **Transacoes legitimas** | 284.315 (99.827%) |
| **Features** | 30 (Time, V1-V28, Amount) |
| **Target** | Class (0=Legitima, 1=Fraude) |
| **Fonte** | [download.tensorflow.org](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) |

> **Nota:** As features V1-V28 sao o resultado de uma transformacao PCA (Principal Component Analysis) aplicada aos dados originais para proteger a privacidade dos clientes.

---

## Tecnologias Utilizadas

| Biblioteca | Versao | Proposito |
|---|---|---|
| [Python](https://www.python.org/) | 3.9+ | Linguagem principal |
| [pandas](https://pandas.pydata.org/) | 2.0+ | Manipulacao de dados |
| [NumPy](https://numpy.org/) | 1.24+ | Computacao numerica |
| [scikit-learn](https://scikit-learn.org/) | 1.3+ | Modelos e metricas ML |
| [Matplotlib](https://matplotlib.org/) | 3.7+ | Visualizacoes |
| [Seaborn](https://seaborn.pydata.org/) | 0.12+ | Visualizacoes estatisticas |
| [joblib](https://joblib.readthedocs.io/) | 1.3+ | Serializacao de modelos |

---

## Estrutura do Projeto

```
fraud-detection-project/
|
|-- main.py                          # Pipeline completo (treinamento)
|-- inspecionar_modelo.py            # Visualizacao dos resultados
|-- README.md                        # Este arquivo
|-- requirements.txt                 # Dependencias
|
|-- plots/                           # Visualizacoes geradas
|   |-- 01_eda_basico.png
|   |-- 02_correlacao.png
|   |-- 03_resultados_finais.png
|   |-- comparacao_modelos.png
|   |-- comparacao_roc_auc.png
|   |-- importancia_features.png
|
|-- modelo_randomforest.joblib       # Modelo treinado (exemplo)
|-- scaler.joblib                    # Scaler ajustado
|-- resultados_modelos.joblib        # Resultados comparativos
|-- creditcard.csv                   # Dataset (baixado automaticamente)
```

---

## Instalacao

### 1. Clone o repositorio

```bash
git clone https://github.com/KelvinOliveiraCode/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependencias

```bash
pip install -r requirements.txt
```

> **Opcional:** Para tecnicas avancadas de balanceamento (SMOTE):
> ```bash
> pip install imbalanced-learn
> ```

---

## Como Usar

### Treinar o modelo

Execute o pipeline completo de treinamento:

```bash
python main.py
```

O script ira:
1. Baixar o dataset (se nao existir localmente)
2. Realizar analise exploratoria (EDA)
3. Pre-processar os dados
4. Balancear as classes
5. Treinar e comparar 3 modelos
6. Selecionar o melhor modelo
7. Salvar artefatos (.joblib)
8. Gerar visualizacoes

### Visualizar resultados

Apos o treinamento, inspecione os modelos salvos:

```bash
python inspecionar_modelo.py
```

Este script exibe:
- Informacoes detalhadas do modelo vencedor
- Tabela comparativa de todos os modelos
- Importancia das features (com grafico)
- Melhores hiperparametros

---

## Passo a Passo do Desenvolvimento

### Fase 1: Entendimento do Problema

O dataset apresenta um **desbalanceamento extremo**: apenas 0.17% das transacoes sao fraudulentas. Isso significa que um modelo que sempre preve "legitima" teria 99.8% de accuracy, mas 0% de recall para fraudes — o que e inutil.

**Metrica principal:** Recall (detectar o maximo de fraudes possivel)
**Metrica secundaria:** F1-Score (equilibrio entre precision e recall)

### Fase 2: Analise Exploratoria (EDA)

Realizada no modulo `explorar_dados()`:

- **Distribuicao das classes**: Visualizacao do desbalanceamento
- **Matriz de correlacao**: Identificacao de features relevantes
- **Analise de Amount**: Comparacao de valores entre classes
- **Features V1-V28**: Histogramas das mais correlacionadas com fraude

Principais insights:
- Features **V17, V14, V12, V10, V16** sao as mais correlacionadas com fraude
- Transacoes fraudulentas tendem a ter valores (Amount) mais baixos
- Distribuicao temporal mostra picos de fraude em horarios especificos

### Fase 3: Pre-processamento

Modulo `preprocessar()`:

1. **Remocao de duplicatas**: 1.081 linhas duplicadas removidas
2. **Divisao estratificada**: 80% treino / 20% teste
3. **RobustScaler**: Aplicado em `Amount` e `Time`
   - Resistente a outliers (usa mediana e IQR)
   - Evita vazamento de dados (fit apenas no treino)

### Fase 4: Balanceamento

Modulo `balancear_dados()`:

Tecnica utilizada: **SMOTE** (Synthetic Minority Over-sampling Technique)
- Gera amostras sinteticas da classe minoritaria
- Cria ~227.000 novas amostras de fraude
- Resultado: dataset balanceado 50/50

> Se `imbalanced-learn` nao estiver instalado, usa `class_weight='balanced'` como fallback.

### Fase 5: Modelagem

Tres modelos treinados com `RandomizedSearchCV`:

| Modelo | Caracteristicas |
|---|---|
| **LogisticRegression** | Rapido, interpretavel, baseline solido |
| **RandomForest** | Ensemble robusto, bom com outliers |
| **ExtraTrees** | Variacao do RF, mais rapido, menos overfitting |

**Otimizacao:**
- `n_iter=5` combinacoes de hiperparametros
- `StratifiedKFold` com 3 folds
- Scoring: `f1` (otimiza equilibrio precision/recall)

### Fase 6: Avaliacao e Selecao

**Criterio de selecao ponderado:**

```
Score = Recall*0.4 + F1-Score*0.3 + Precision*0.2 + Accuracy*0.1
```

| Prioridade | Peso | Justificativa |
|---|---|---|
| Recall | 40% | Nao perder fraudes e crucial |
| F1-Score | 30% | Equilibrio geral |
| Precision | 20% | Evitar falsos alarmes |
| Accuracy | 10% | Menos relevante em dados desbalanceados |

### Fase 7: Salvamento

Artefatos gerados:
- `modelo_*.joblib` — Modelo treinado pronto para producao
- `scaler.joblib` — Scaler para novas transacoes
- `resultados_modelos.joblib` — Metricas comparativas

---

## Resultados

Exemplo de resultado tipico (pode variar conforme a execucao):

### Melhor Modelo: RandomForest

```
Accuracy:         0.9994
Precision:        0.8046
Recall:           0.7368
F1-Score:         0.8046
ROC AUC:          0.9445

Melhores hiperparametros:
  class_weight: balanced
  max_depth: 10
  min_samples_split: 2
  n_estimators: 50
```

### Comparacao de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.9752 | 0.0562 | **0.8737** | 0.1057 | 0.9657 |
| RandomForest | **0.9994** | **0.8046** | 0.7368 | **0.8046** | 0.9445 |
| ExtraTrees | 0.9987 | 0.6667 | 0.8000 | 0.6667 | **0.9621** |

> **RandomForest** foi selecionado por apresentar o melhor equilibrio ponderado entre as metricas.

---

## Prints do Projeto

### Distribuicao das Classes
![Distribuicao](plots/01_eda_basico.png)

### Matriz de Correlacao
![Correlacao](plots/02_correlacao.png)

### Resultados Finais
![Resultados](plots/03_resultados_finais.png)

---

## Autor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/KelvinOliveiraCode">
        <img src="https://github.com/KelvinOliveiraCode.png" width="100px;" alt="Kelvin Oliveira"/><br>
        <sub><b>Kelvin Oliveira</b></sub>
      </a>
    </td>
  </tr>
</table>

- **GitHub**: [@KelvinOliveiraCode](https://github.com/KelvinOliveiraCode)
- **LinkedIn**: [kelvin-oliveira](https://www.linkedin.com/in/kelvin-oliveira/)
- **Plataforma**: [DIO - Digital Innovation One](https://www.dio.me/)

---

## Licenca

Este projeto esta licenciado sob a licenca MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## Agradecimentos

- [DIO](https://www.dio.me/) pela oportunidade de aprendizado
- [scikit-learn](https://scikit-learn.org/) pela excelente documentacao
- [Kaggle](https://www.kaggle.com/) pelo dataset

---

<p align="center">
  Desenvolvido com <span style="color: #e74c3c;">&#10084;</span> para o curso da DIO
</p>
