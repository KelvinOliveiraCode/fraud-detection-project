"""
================================================================================
INSPECIONADOR DE MODELOS - Detector de Fraudes Bancarias
================================================================================
Script para visualizar informacoes dos arquivos .joblib gerados pelo treinamento.

Como usar:
    1. Salve este arquivo na MESMA pasta do projeto (onde estao os .joblib)
    2. Execute: python inspecionar_modelo.py

Requisitos:
    pip install pandas numpy matplotlib seaborn joblib scikit-learn
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def listar_arquivos():
    """Lista todos os arquivos .joblib na pasta atual."""
    arquivos = glob.glob("*.joblib")
    if not arquivos:
        print("[!] Nenhum arquivo .joblib encontrado na pasta atual.")
        print("    Certifique-se de executar este script na pasta do projeto.")
        return None
    print(f"\n[*] Arquivos .joblib encontrados: {len(arquivos)}")
    for i, arq in enumerate(sorted(arquivos), 1):
        tamanho = os.path.getsize(arq) / 1024
        print(f"    {i}. {arq} ({tamanho:.1f} KB)")
    return sorted(arquivos)


def ver_modelo(caminho):
    """Exibe informacoes detalhadas do modelo treinado."""
    print(f"\n{'='*70}")
    print(f"  MODELO: {caminho}")
    print(f"{'='*70}")

    modelo = joblib.load(caminho)
    tipo = type(modelo).__name__
    modulo = type(modelo).__module__

    print(f"\n  Tipo:        {tipo}")
    print(f"  Modulo:      {modulo}")

    # Hiperparametros
    if hasattr(modelo, 'get_params'):
        print(f"\n  {'-'*66}")
        print("  HIPERPARAMETROS")
        print(f"  {'-'*66}")
        params = modelo.get_params()
        for chave, valor in sorted(params.items()):
            print(f"    {chave:30s} : {valor}")

    # Features
    if hasattr(modelo, 'n_features_in_'):
        print(f"\n  Numero de features de entrada: {modelo.n_features_in_}")

    if hasattr(modelo, 'feature_names_in_'):
        print(f"  Nomes das features: {list(modelo.feature_names_in_)}")

    # Classes
    if hasattr(modelo, 'classes_'):
        print(f"  Classes: {modelo.classes_}")
        print(f"    0 = Legitima | 1 = Fraude")

    # Importancia das features
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        n = len(importancias)

        # Gerar nomes de features
        if n == 30:
            nomes = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        elif n == 31:
            nomes = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Amount_per_hour']
        else:
            nomes = [f'Feature_{i}' for i in range(n)]

        print(f"\n  {'-'*66}")
        print("  IMPORTANCIA DAS FEATURES (Top 15)")
        print(f"  {'-'*66}")

        ranking = sorted(zip(nomes, importancias), key=lambda x: x[1], reverse=True)
        for nome, imp in ranking[:15]:
            barra = '█' * int(imp * 100)
            print(f"    {nome:20s} : {imp:.4f} {barra}")

        # Plot de importancia
        plt.figure(figsize=(10, 8))
        top_nomes = [r[0] for r in ranking[:15]]
        top_imps = [r[1] for r in ranking[:15]]
        cores = ['#e74c3c' if 'Amount' in n or 'Time' in n else '#3498db' for n in top_nomes]
        plt.barh(range(len(top_nomes)), top_imps, color=cores)
        plt.yticks(range(len(top_nomes)), top_nomes)
        plt.xlabel('Importancia')
        plt.title(f'Importancia das Features - {tipo}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('plots/importancia_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  [*] Plot salvo: plots/importancia_features.png")

    # Outros atributos uteis
    print(f"\n  {'-'*66}")
    print("  OUTROS ATRIBUTOS")
    print(f"  {'-'*66}")

    atributos = ['n_outputs_', 'n_classes_', 'max_features_', 'n_estimators',
                 'tree_', 'coef_', 'intercept_', 'classes_']
    for attr in atributos:
        if hasattr(modelo, attr):
            valor = getattr(modelo, attr)
            if attr == 'coef_':
                print(f"    {attr:20s} : shape {valor.shape}")
            elif attr == 'intercept_':
                print(f"    {attr:20s} : {valor}")
            else:
                print(f"    {attr:20s} : {valor}")

    return modelo


def ver_scaler(caminho):
    """Exibe informacoes do scaler."""
    print(f"\n{'='*70}")
    print(f"  SCALER: {caminho}")
    print(f"{'='*70}")

    scaler = joblib.load(caminho)
    tipo = type(scaler).__name__

    print(f"\n  Tipo: {tipo}")

    if hasattr(scaler, 'center_'):
        print(f"\n  Centros (medianas):")
        for i, c in enumerate(scaler.center_):
            print(f"    Feature {i}: {c:.6f}")

    if hasattr(scaler, 'scale_'):
        print(f"\n  Escalas (IQR):")
        for i, s in enumerate(scaler.scale_):
            print(f"    Feature {i}: {s:.6f}")

    if hasattr(scaler, 'data_min_'):
        print(f"\n  Minimos:")
        for i, m in enumerate(scaler.data_min_):
            print(f"    Feature {i}: {m:.6f}")

    if hasattr(scaler, 'data_max_'):
        print(f"\n  Maximos:")
        for i, m in enumerate(scaler.data_max_):
            print(f"    Feature {i}: {m:.6f}")

    return scaler


def ver_resultados(caminho):
    """Exibe tabela comparativa e graficos dos resultados dos modelos."""
    print(f"\n{'='*70}")
    print(f"  RESULTADOS DOS MODELOS: {caminho}")
    print(f"{'='*70}")

    resultados = joblib.load(caminho)

    print(f"\n  Modelos treinados: {len(resultados)}")
    print(f"  Nomes: {', '.join(resultados.keys())}")

    # Criar DataFrame comparativo
    dados = []
    for nome, r in resultados.items():
        dados.append({
            'Modelo': nome,
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score'],
            'ROC AUC': r['roc_auc'],
            'Tempo (s)': r['tempo']
        })

    df = pd.DataFrame(dados)

    print(f"\n  {'-'*66}")
    print("  TABELA COMPARATIVA")
    print(f"  {'-'*66}")
    print(df.to_string(index=False, float_format='%.4f'))

    # Destacar melhor modelo
    df['Score'] = df['Recall'] * 0.4 + df['F1-Score'] * 0.3 + df['Precision'] * 0.2 + df['Accuracy'] * 0.1
    melhor = df.loc[df['Score'].idxmax(), 'Modelo']
    print(f"\n  [*] Melhor modelo (por ponderacao): {melhor}")

    # Grafico comparativo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cores = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for idx, (metrica, cor) in enumerate(zip(metricas, cores)):
        ax = axes[idx // 2, idx % 2]
        barras = ax.bar(df['Modelo'], df[metrica], color=cor, edgecolor='black')
        ax.set_title(metrica, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Valor')

        # Destacar melhor
        melhor_idx = df[metrica].idxmax()
        barras[melhor_idx].set_color('#e74c3c')

        for i, v in enumerate(df[metrica]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.suptitle('Comparacao de Modelos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/comparacao_modelos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [*] Plot salvo: plots/comparacao_modelos.png")

    # ROC AUC comparativo
    plt.figure(figsize=(8, 6))
    plt.bar(df['Modelo'], df['ROC AUC'], color='#9b59b6', edgecolor='black')
    plt.title('ROC AUC por Modelo', fontsize=14, fontweight='bold')
    plt.ylabel('ROC AUC')
    plt.ylim(0.5, 1.05)
    for i, v in enumerate(df['ROC AUC']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comparacao_roc_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [*] Plot salvo: plots/comparacao_roc_auc.png")

    # Melhores hiperparametros
    print(f"\n  {'-'*66}")
    print("  MELHORES HIPERPARAMETROS POR MODELO")
    print(f"  {'-'*66}")
    for nome, r in resultados.items():
        print(f"\n    {nome}:")
        for param, valor in r['best_params'].items():
            print(f"      {param:25s} : {valor}")

    return resultados


def main():
    """Funcao principal que executa a inspecao completa."""
    print(f"\n{'#'*70}")
    print("# INSPECIONADOR DE MODELOS .JOBLIB")
    print("# Detector de Fraudes Bancarias")
    print(f"{'#'*70}")

    # Criar pasta de plots
    os.makedirs('plots', exist_ok=True)

    # Listar arquivos
    arquivos = listar_arquivos()
    if not arquivos:
        return

    # Processar cada arquivo
    for arquivo in arquivos:
        nome_lower = arquivo.lower()

        if 'scaler' in nome_lower:
            ver_scaler(arquivo)
        elif 'resultado' in nome_lower:
            ver_resultados(arquivo)
        else:
            ver_modelo(arquivo)

    # Resumo final
    print(f"\n{'='*70}")
    print("  RESUMO DA INSPECAO")
    print(f"{'='*70}")
    print(f"  Total de arquivos analisados: {len(arquivos)}")
    print(f"  Plots gerados em: plots/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
