"""
Detector de Fraudes Bancarias - Credit Card Fraud Detection
Versao OTIMIZADA para execucao rapida

Dataset: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Requisitos:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib

Como executar:
    python main.py
"""

import os
import warnings
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")
np.random.seed(42)
pd.set_option("display.max_columns", 50)

MATPLOTLIB_VERSION = tuple(map(int, matplotlib.__version__.split(".")[:2]))

# =============================================================================
# 1. CARREGAR DADOS
# =============================================================================

def carregar_dados(url="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"):
    """Carrega o dataset de fraudes de cartao de credito."""
    arquivo_local = "creditcard.csv"

    if os.path.exists(arquivo_local):
        print(f"[*] Carregando dataset local: {arquivo_local}")
        df = pd.read_csv(arquivo_local)
    else:
        print(f"[*] Baixando dataset...")
        df = pd.read_csv(url)
        df.to_csv(arquivo_local, index=False)
        print(f"[*] Dataset salvo como: {arquivo_local}")

    print(f"\n{'='*60}")
    print("INFORMACOES GERAIS DO DATASET")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print(f"\n{'-'*60}")
    print("VALORES AUSENTES / DUPLICATAS")
    print(f"{'-'*60}")
    print(f"Valores ausentes: {df.isnull().sum().sum()}")
    print(f"Duplicatas: {df.duplicated().sum()}")

    print(f"\n{'-'*60}")
    print("DISTRIBUICAO DA VARIAVEL ALVO (Class)")
    print(f"{'-'*60}")
    class_counts = df["Class"].value_counts()
    class_pct = df["Class"].value_counts(normalize=True) * 100
    print(f"Legitima (0): {class_counts[0]:,} ({class_pct[0]:.4f}%)")
    print(f"Fraude (1):   {class_counts[1]:,} ({class_pct[1]:.4f}%)")
    print(f"Razao: {class_counts[0]/class_counts[1]:.0f}:1")

    return df


# =============================================================================
# 2. ANALISE EXPLORATORIA
# =============================================================================

def explorar_dados(df):
    """Analise exploratoria rapida com plots essenciais."""
    print(f"\n{'='*60}")
    print("ANALISE EXPLORATORIA (EDA)")
    print(f"{'='*60}")

    os.makedirs("plots", exist_ok=True)

    # Distribuicao das classes + Amount
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = df["Class"].value_counts()

    axes[0].bar(["Legitima (0)", "Fraude (1)"], class_counts.values, 
                color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[0].set_title("Distribuicao das Classes", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Contagem")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5000, f"{v:,}", ha="center", fontsize=12, fontweight="bold")

    axes[1].hist(df[df["Class"]==0]["Amount"], bins=50, alpha=0.7, label="Legitima", 
                 color="#2ecc71", density=True)
    axes[1].hist(df[df["Class"]==1]["Amount"], bins=50, alpha=0.7, label="Fraude", 
                 color="#e74c3c", density=True)
    axes[1].set_title("Amount por Classe", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Densidade")
    axes[1].legend()
    axes[1].set_xlim(0, 1000)

    plt.tight_layout()
    plt.savefig("plots/01_eda_basico.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Features mais correlacionadas
    corr_matrix = df.corr()
    class_corr = corr_matrix["Class"].drop("Class").sort_values(key=abs, ascending=False)
    print("\nTop 10 features mais correlacionadas com Class:")
    print(class_corr.head(10).round(4).to_string())

    # Matriz de correlacao
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r", center=0, 
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Matriz de Correlacao", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/02_correlacao.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\n[*] Plots salvos em 'plots/'")

    return df


# =============================================================================
# 3. PRE-PROCESSAMENTO
# =============================================================================

def preprocessar(df):
    """Pre-processamento: remove duplicatas, split, scale."""
    print(f"\n{'='*60}")
    print("PRE-PROCESSAMENTO")
    print(f"{'='*60}")

    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"[*] Duplicatas removidas: {df.shape[0] - df_clean.shape[0]}")

    X = df_clean.drop("Class", axis=1)
    y = df_clean["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'-'*60}")
    print("DIVISAO DOS DADOS")
    print(f"{'-'*60}")
    print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
    print(f"Fraudes treino: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"Fraudes teste:  {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test_scaled[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    print(f"[*] RobustScaler aplicado em Amount e Time")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# 4. BALANCEAMENTO
# =============================================================================

def balancear_dados(X_train, y_train):
    """Tenta SMOTE; se nao disponivel, usa class_weight nos modelos."""
    print(f"\n{'='*60}")
    print("BALANCEAMENTO")
    print(f"{'='*60}")

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"[*] SMOTE aplicado: {len(y_res):,} amostras (antes: {len(y_train):,})")
        print(f"[*] Distribuicao: {dict(pd.Series(y_res).value_counts().sort_index())}")
        return X_res, y_res, "SMOTE"
    except ImportError:
        print("[!] imbalanced-learn nao instalado.")
        print("[*] Usando class_weight='balanced' nos modelos.")
        return X_train, y_train, "class_weight"


# =============================================================================
# 5. MODELAGEM (OTIMIZADA)
# =============================================================================

def treinar_modelos(X_train, y_train, X_test, y_test):
    """Treina modelos leves e rapidos."""
    print(f"\n{'='*60}")
    print("MODELAGEM E TREINAMENTO")
    print(f"{'='*60}")
    print("[*] Modo rapido: modelos leves, poucas combinacoes")

    # Verificar se usamos class_weight (quando SMOTE nao esta disponivel)
    usar_class_weight = (len(np.unique(y_train)) == 2 and 
                        np.bincount(y_train)[0] != np.bincount(y_train)[1])

    modelos_config = {
        "LogisticRegression": {
            "modelo": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
            "params": {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced"] if usar_class_weight else [None]
            }
        },
        "RandomForest": {
            "modelo": RandomForestClassifier(
                n_estimators=30,
                max_depth=10,
                random_state=42, 
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [30, 50],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced"] if usar_class_weight else [None]
            }
        },
        "ExtraTrees": {
            "modelo": ExtraTreesClassifier(
                n_estimators=30,
                max_depth=10,
                random_state=42, 
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [30, 50],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced"] if usar_class_weight else [None]
            }
        }
    }

    resultados = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for nome, config in modelos_config.items():
        print(f"\n{'-'*60}")
        print(f"Treinando: {nome}")
        print(f"{'-'*60}")

        inicio = time.time()

        search = RandomizedSearchCV(
            estimator=config["modelo"],
            param_distributions=config["params"],
            n_iter=5,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        y_prob = search.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        tempo = time.time() - inicio

        resultados[nome] = {
            "modelo": search.best_estimator_,
            "best_params": search.best_params_,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "tempo": tempo
        }

        print(f"  Params: {search.best_params_}")
        print(f"  Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC: {roc:.4f}")
        print(f"  Tempo: {tempo:.1f}s")

    return resultados


# =============================================================================
# 6. AVALIACAO
# =============================================================================

def avaliar_modelo(nome, y_test, y_pred, y_prob):
    """Exibe metricas detalhadas."""
    print(f"\n{'='*60}")
    print(f"AVALIACAO: {nome}")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitima", "Fraude"], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predito")
    print(f"                 Legitima  Fraude")
    print(f"Real  Legitima   {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"      Fraude     {cm[1,0]:6d}    {cm[1,1]:6d}")

    roc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {roc:.4f}")

    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    print(f"Specificity: {specificity:.4f}")


# =============================================================================
# 7. ESCOLHA DO MELHOR MODELO
# =============================================================================

def escolher_melhor_modelo(resultados):
    """Seleciona o melhor modelo pelo equilibrio ponderado."""
    print(f"\n{'='*60}")
    print("COMPARACAO DE MODELOS")
    print(f"{'='*60}")

    comparacao = pd.DataFrame({
        "Modelo": list(resultados.keys()),
        "Accuracy": [r["accuracy"] for r in resultados.values()],
        "Precision": [r["precision"] for r in resultados.values()],
        "Recall": [r["recall"] for r in resultados.values()],
        "F1-Score": [r["f1_score"] for r in resultados.values()],
        "ROC_AUC": [r["roc_auc"] for r in resultados.values()],
        "Tempo(s)": [r["tempo"] for r in resultados.values()]
    })

    print("\nTabela Comparativa:")
    print(comparacao.to_string(index=False, float_format="%.4f"))

    comparacao["score"] = (
        comparacao["Recall"] * 0.4 +
        comparacao["F1-Score"] * 0.3 +
        comparacao["Precision"] * 0.2 +
        comparacao["Accuracy"] * 0.1
    )

    melhor_idx = comparacao["score"].idxmax()
    melhor_nome = comparacao.loc[melhor_idx, "Modelo"]

    print(f"\n{'-'*60}")
    print("CRITERIO DE SELECAO")
    print(f"{'-'*60}")
    print("Ponderacao: Recall(40%) + F1(30%) + Precision(20%) + Accuracy(10%)")

    return melhor_nome, resultados[melhor_nome]


# =============================================================================
# 8. SALVAMENTO
# =============================================================================

def salvar_modelo(nome, modelo, scaler, resultados):
    """Salva o modelo treinado e scaler."""
    print(f"\n{'='*60}")
    print("SALVAMENTO")
    print(f"{'='*60}")

    modelo_path = f"modelo_{nome.lower().replace(' ', '_')}.joblib"
    joblib.dump(modelo, modelo_path)
    print(f"[*] Modelo: {modelo_path}")

    scaler_path = "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[*] Scaler: {scaler_path}")

    resultados_path = "resultados_modelos.joblib"
    joblib.dump(resultados, resultados_path)
    print(f"[*] Resultados: {resultados_path}")

    return modelo_path, scaler_path


# =============================================================================
# 9. VISUALIZACAO FINAL
# =============================================================================

def plotar_resultados_finais(melhor_nome, melhor_resultado, y_test):
    """Gera plots finais do melhor modelo."""
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, melhor_resultado["y_prob"])
    roc_auc = roc_auc_score(y_test, melhor_resultado["y_prob"])

    axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, 
                 label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC - {melhor_nome}", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_test, melhor_resultado["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Legitima", "Fraude"],
                yticklabels=["Legitima", "Fraude"])
    axes[1].set_title(f"Confusion Matrix - {melhor_nome}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Real")

    plt.tight_layout()
    plt.savefig("plots/03_resultados_finais.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[*] Plot final salvo: plots/03_resultados_finais.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Pipeline principal."""
    inicio_total = time.time()

    print(f"\n{'#'*60}")
    print("# DETECTOR DE FRAUDES BANCARIAS - MODO RAPIDO")
    print(f"{'#'*60}")
    print(f"[*] Matplotlib: {matplotlib.__version__}")

    # 1. Carregar dados
    df = carregar_dados()

    # 2. EDA
    df = explorar_dados(df)

    # 3. Pre-processamento
    X_train, X_test, y_train, y_test, scaler = preprocessar(df)

    # 4. Balanceamento
    X_train_bal, y_train_bal, tecnica = balancear_dados(X_train, y_train)

    # 5. Modelagem
    resultados = treinar_modelos(X_train_bal, y_train_bal, X_test, y_test)

    # 6. Escolher melhor
    melhor_nome, melhor_resultado = escolher_melhor_modelo(resultados)

    # 7. Avaliar
    avaliar_modelo(melhor_nome, y_test, melhor_resultado["y_pred"], melhor_resultado["y_prob"])

    # 8. Plotar
    plotar_resultados_finais(melhor_nome, melhor_resultado, y_test)

    # 9. Salvar
    salvar_modelo(melhor_nome, melhor_resultado["modelo"], scaler, resultados)

    # 10. Resultado final
    tempo_total = time.time() - inicio_total

    print(f"\n{'='*60}")
    print("MELHOR MODELO ENCONTRADO")
    print(f"{'='*60}")
    print(f"\nModelo:           {melhor_nome}")
    print(f"Accuracy:         {melhor_resultado['accuracy']:.4f}")
    print(f"Precision:        {melhor_resultado['precision']:.4f}")
    print(f"Recall:           {melhor_resultado['recall']:.4f}")
    print(f"F1-Score:         {melhor_resultado['f1_score']:.4f}")
    print(f"ROC AUC:          {melhor_resultado['roc_auc']:.4f}")
    print(f"\nMelhores hiperparametros:")
    for param, valor in melhor_resultado["best_params"].items():
        print(f"  {param}: {valor}")
    print(f"\nBalanceamento: {tecnica}")
    print(f"Tempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min)")
    print(f"{'='*60}")

    print("\nClassification Report Completo:")
    print(classification_report(y_test, melhor_resultado["y_pred"], 
                                target_names=["Legitima", "Fraude"], digits=4))

    return melhor_resultado["modelo"], scaler


if __name__ == "__main__":
    modelo, scaler = main()
