"""
Script para gerar relatório do modelo treinado
"""
import pickle
import pandas as pd
from datetime import datetime

print("="*80)
print("RELATÓRIO DO MODELO IBOVESPA")
print("="*80)

# Carregar modelo
try:
    with open('melhor_modelo_ibovespa_FINAL.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n>> Modelo carregado com sucesso!")
    print(f"\n{'='*80}")
    print(f"INFORMAÇÕES DO MODELO")
    print(f"{'='*80}")
    
    print(f"\n   Estratégia: {model_data.get('strategy', model_data.get('model_name', 'N/A'))}")
    print(f"   Acurácia: {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)")
    print(f"   Janela de teste: {model_data['test_size']} dias")
    
    if 'threshold' in model_data:
        print(f"   Threshold otimizado: {model_data['threshold']:.2f}")
    
    if 'config' in model_data:
        print(f"\n   Configuração:")
        for key, value in model_data['config'].items():
            print(f"      - {key}: {value}")
    
    print(f"\n{'='*80}")
    
    if model_data['accuracy'] >= 0.75:
        print(">>> META DE 75% ATINGIDA!")
        print(">>> MODELO APROVADO PARA PRODUÇÃO!")
    else:
        diff = (0.75 - model_data['accuracy']) * 100
        print(f"> Faltam {diff:.2f}% para atingir a meta de 75%")
        print(f"> Sugestão: Executar novamente ou adicionar features externas")
    
    print(f"{'='*80}")
    
    # Informações dos dados
    print(f"\n{'='*80}")
    print(f"DADOS UTILIZADOS")
    print(f"{'='*80}")
    
    df = pd.read_csv('data/Dados Históricos - Ibovespa.csv')
    print(f"\n   Arquivo: data/Dados Históricos - Ibovespa.csv")
    print(f"   Registros: {len(df)}")
    print(f"   Features criadas: {len(model_data.get('features', []))} indicadores técnicos")
    
    df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
    print(f"   Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
    
    anos = (df['Data'].max() - df['Data'].min()).days / 365.25
    print(f"   Duração: {anos:.1f} anos")
    
    print(f"\n{'='*80}")
    print(f"Relatório gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    print(f"{'='*80}\n")
    
except FileNotFoundError:
    print("\n>> ERRO: Modelo não encontrado!")
    print("   Execute primeiro: python3 ibovespa_ATINGIR_75.py")
    print()
except Exception as e:
    print(f"\n>> ERRO ao carregar modelo: {e}\n")

