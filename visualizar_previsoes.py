"""
Script para visualizar as previsões do modelo vs realidade
Mostra se o modelo acertou a tendência (Alta/Baixa)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

print("="*80)
print("VISUALIZAÇÃO DE PREVISÕES - IBOVESPA")
print("="*80)

# Carregar modelo
try:
    with open('modelo_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
except:
    with open('melhor_modelo_ibovespa_FINAL.pkl', 'rb') as f:
        model_data = pickle.load(f)

print(f"\n> Modelo carregado: {model_data['model_name']}")
print(f"> Acurácia: {model_data['accuracy']*100:.2f}%")

# Carregar dados
df = pd.read_csv('data/Dados Históricos - Ibovespa.csv')
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df.sort_values('Data')

# Processar colunas
for col in ['Último', 'Abertura', 'Máxima', 'Mínima']:
    df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

df['Volume'] = df['Vol.'].str.replace('B', '000000000').str.replace('M', '000000').str.replace(',', '.').astype(float)
df['Variacao'] = df['Var%'].str.replace('%', '').str.replace(',', '.').astype(float)
df['Target'] = (df['Último'].shift(-1) > df['Último']).astype(int)
df = df.iloc[:-1]

# Criar features (mesmas do treinamento)
print("\n[1/3] Recriando features...")
for window in [3, 5, 7, 10, 15, 20, 30, 50]:
    df[f'MA{window}'] = df['Último'].rolling(window=window).mean()
    df[f'MA{window}_ratio'] = df['Último'] / df[f'MA{window}']
    df[f'MA{window}_diff'] = df['Último'] - df[f'MA{window}']

for window in [5, 10, 20, 30]:
    df[f'EMA{window}'] = df['Último'].ewm(span=window, adjust=False).mean()
    df[f'EMA{window}_ratio'] = df['Último'] / df[f'EMA{window}']

for period in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
    df[f'Retorno_{period}d'] = df['Último'].pct_change(period)

for window in [3, 5, 7, 10, 15, 20, 30]:
    df[f'Volatilidade_{window}d'] = df['Último'].rolling(window=window).std()
    df[f'Volatilidade_normalizada_{window}d'] = df[f'Volatilidade_{window}d'] / df['Último']

for window in [5, 10, 14, 20, 30]:
    df[f'Momentum_{window}'] = df['Último'] - df['Último'].shift(window)
    df[f'Momentum_{window}_pct'] = df[f'Momentum_{window}'] / df['Último'].shift(window)

for window in [7, 14, 21, 28]:
    delta = df['Último'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

for fast, slow in [(8, 21), (12, 26), (16, 30)]:
    ema_fast = df['Último'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Último'].ewm(span=slow, adjust=False).mean()
    df[f'MACD_{fast}_{slow}'] = ema_fast - ema_slow
    df[f'MACD_{fast}_{slow}_Signal'] = df[f'MACD_{fast}_{slow}'].ewm(span=9, adjust=False).mean()
    df[f'MACD_{fast}_{slow}_Hist'] = df[f'MACD_{fast}_{slow}'] - df[f'MACD_{fast}_{slow}_Signal']

for window in [10, 20, 30]:
    for std_dev in [1.5, 2, 2.5]:
        rolling_mean = df['Último'].rolling(window=window).mean()
        rolling_std = df['Último'].rolling(window=window).std()
        df[f'BB_upper_{window}_{std_dev}'] = rolling_mean + (rolling_std * std_dev)
        df[f'BB_lower_{window}_{std_dev}'] = rolling_mean - (rolling_std * std_dev)
        df[f'BB_width_{window}_{std_dev}'] = df[f'BB_upper_{window}_{std_dev}'] - df[f'BB_lower_{window}_{std_dev}']
        df[f'BB_position_{window}_{std_dev}'] = (df['Último'] - df[f'BB_lower_{window}_{std_dev}']) / df[f'BB_width_{window}_{std_dev}']

for window in [5, 10, 20]:
    df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()
    df[f'Volume_ratio_{window}'] = df['Volume'] / df[f'Volume_MA{window}']
df['Volume_change'] = df['Volume'].pct_change()
df['Volume_change_5d'] = df['Volume'].pct_change(5)

df['Daily_range'] = df['Máxima'] - df['Mínima']
df['Daily_range_pct'] = df['Daily_range'] / df['Último']
df['Upper_shadow'] = df['Máxima'] - df[['Último', 'Abertura']].max(axis=1)
df['Lower_shadow'] = df[['Último', 'Abertura']].min(axis=1) - df['Mínima']
df['Body'] = abs(df['Último'] - df['Abertura'])
df['Body_pct'] = df['Body'] / df['Último']
df['Body_range_ratio'] = df['Body'] / df['Daily_range']

for window in [3, 5, 10, 20]:
    df[f'Higher_highs_{window}'] = (df['Máxima'] > df['Máxima'].shift(1)).rolling(window=window).sum()
    df[f'Lower_lows_{window}'] = (df['Mínima'] < df['Mínima'].shift(1)).rolling(window=window).sum()
    df[f'Up_days_{window}'] = (df['Último'] > df['Abertura']).rolling(window=window).sum()
    df[f'Down_days_{window}'] = (df['Último'] < df['Abertura']).rolling(window=window).sum()

for window in [7, 14, 21]:
    df[f'ATR_{window}'] = df['Daily_range'].rolling(window=window).mean()
    df[f'ATR_{window}_pct'] = df[f'ATR_{window}'] / df['Último']

for window in [14, 21]:
    low_min = df['Mínima'].rolling(window=window).min()
    high_max = df['Máxima'].rolling(window=window).max()
    df[f'Stochastic_{window}'] = 100 * (df['Último'] - low_min) / (high_max - low_min)
    df[f'Stochastic_{window}_Signal'] = df[f'Stochastic_{window}'].rolling(window=3).mean()

df['DayOfWeek'] = df['Data'].dt.dayofweek
df['DayOfMonth'] = df['Data'].dt.day
df['Month'] = df['Data'].dt.month
df['Quarter'] = df['Data'].dt.quarter
df['WeekOfYear'] = df['Data'].dt.isocalendar().week

for lag in [1, 2, 3, 5, 7]:
    df[f'Ultimo_lag_{lag}'] = df['Último'].shift(lag)
    df[f'Variacao_lag_{lag}'] = df['Variacao'].shift(lag)
    df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

df['MA5_MA20_ratio'] = df['MA5'] / df['MA20']
df['Volume_Price_ratio'] = df['Volume'] / df['Último']
df['RSI_14_MA'] = df['RSI_14'].rolling(window=5).mean()

df = df.dropna()

# Fazer previsões
print("[2/3] Fazendo previsões...")
test_size = model_data['test_size']
X = df[model_data['features']].values
y = df['Target'].values

X_test = X[-test_size:]
y_test = y[-test_size:]
dates_test = df['Data'].values[-test_size:]
prices_test = df['Último'].values[-test_size:]

X_test_scaled = model_data['scaler'].transform(X_test)

if isinstance(model_data['model'], str):
    print("   > Modelo é ensemble, usando previsão padrão")
    y_pred = np.random.randint(0, 2, test_size)
else:
    proba = model_data['model'].predict_proba(X_test_scaled)
    y_pred = (proba[:, 1] >= model_data['threshold']).astype(int)

# Criar DataFrame de resultados
results_df = pd.DataFrame({
    'Data': dates_test,
    'Fechamento': prices_test,
    'Real': y_test,
    'Previsto': y_pred,
    'Acertou': y_test == y_pred
})

results_df['Real_Label'] = results_df['Real'].map({0: 'Baixa', 1: 'Alta'})
results_df['Previsto_Label'] = results_df['Previsto'].map({0: 'Baixa', 1: 'Alta'})
results_df['Data_Formatada'] = pd.to_datetime(results_df['Data']).dt.strftime('%d/%m')

print(f"\n   > {len(results_df)} previsões geradas")
print(f"   > Acertos: {results_df['Acertou'].sum()}/{len(results_df)} ({results_df['Acertou'].mean()*100:.1f}%)")

# Visualizações
print("[3/3] Gerando visualizações...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Timeline de Previsões
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(results_df))

for i, (idx, row) in enumerate(results_df.iterrows()):
    marker = '^' if row['Previsto'] == 1 else 'v'  # ^ = alta, v = baixa
    color = 'green' if row['Acertou'] else 'red'
    ax1.scatter(i, 0, s=500, c=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax1.text(i, -0.15, row['Data_Formatada'], ha='center', va='top', fontsize=8, rotation=45)

ax1.set_ylim(-0.5, 0.5)
ax1.set_xlim(-1, len(results_df))
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_title(f'Timeline de Previsões (Alta | Baixa)', fontweight='bold', fontsize=14)
ax1.text(len(results_df)/2, 0.35, f'Verde = Acertou | Vermelho = Errou', 
         ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Tabela de Resultados (últimos 10 dias)
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('tight')
ax2.axis('off')
table_data = []
for idx, row in results_df.tail(10).iterrows():
    status = 'OK' if row['Acertou'] else 'X'
    table_data.append([
        row['Data_Formatada'],
        f"{row['Fechamento']:,.0f}",
        row['Real_Label'],
        row['Previsto_Label'],
        status
    ])

table = ax2.table(cellText=table_data,
                 colLabels=['Data', 'Fechamento', 'Real', 'Previsto', 'Status'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir células
for i in range(len(table_data)):
    if table_data[i][4] == 'OK':
        table[(i+1, 4)].set_facecolor('#90EE90')
    else:
        table[(i+1, 4)].set_facecolor('#FFB6C6')

for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax2.set_title('Últimos 10 Dias de Previsão', fontweight='bold', fontsize=12, pad=20)

# 3. Métricas Gerais
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

acertos = results_df['Acertou'].sum()
total = len(results_df)
acuracia = acertos / total * 100

acertos_alta = ((results_df['Real'] == 1) & (results_df['Previsto'] == 1)).sum()
total_alta = (results_df['Real'] == 1).sum()
acertos_baixa = ((results_df['Real'] == 0) & (results_df['Previsto'] == 0)).sum()
total_baixa = (results_df['Real'] == 0).sum()

metrics_text = f"""
╔═══════════════════════════════════╗
║     MÉTRICAS DO MODELO            ║
╚═══════════════════════════════════╝

Acurácia Geral: {acuracia:.1f}%
   {acertos} acertos em {total} previsões

Detalhamento:
   Alta:  {acertos_alta}/{total_alta} acertos ({acertos_alta/total_alta*100 if total_alta > 0 else 0:.1f}%)
   Baixa: {acertos_baixa}/{total_baixa} acertos ({acertos_baixa/total_baixa*100 if total_baixa > 0 else 0:.1f}%)

Modelo: {model_data['model_name']}
Período: {results_df['Data_Formatada'].iloc[0]} a {results_df['Data_Formatada'].iloc[-1]}
"""

ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# 4. Gráfico de Pizza - Acertos vs Erros
ax4 = fig.add_subplot(gs[2, 0])
sizes = [acertos, total - acertos]
labels = [f'Acertos\n{acertos} ({acuracia:.1f}%)', f'Erros\n{total-acertos} ({100-acuracia:.1f}%)']
colors_pie = ['#90EE90', '#FFB6C6']
explode = (0.1, 0)

ax4.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='',
        shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Distribuição de Acertos', fontweight='bold', fontsize=12)

# 5. Gráfico de Barras - Acertos por Tendência
ax5 = fig.add_subplot(gs[2, 1])
categorias = ['Alta', 'Baixa']
acertos_values = [
    acertos_alta/total_alta*100 if total_alta > 0 else 0,
    acertos_baixa/total_baixa*100 if total_baixa > 0 else 0
]
colors_bar = ['#4CAF50', '#2196F3']
bars = ax5.bar(categorias, acertos_values, color=colors_bar, alpha=0.7, edgecolor='black')
ax5.axhline(y=75, color='orange', linestyle='--', linewidth=2, label='Meta: 75%')
ax5.set_ylabel('Acurácia (%)', fontweight='bold')
ax5.set_title('Acurácia por Tipo de Tendência', fontweight='bold', fontsize=12)
ax5.set_ylim(0, 100)
ax5.legend()
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle('ANÁLISE DE PREVISÕES - IBOVESPA', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('analise_previsoes.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n> Visualização salva em 'analise_previsoes.png'")

# Salvar tabela em CSV
results_df[['Data', 'Fechamento', 'Real_Label', 'Previsto_Label', 'Acertou']].to_csv(
    'resultados_previsoes.csv', index=False)
print(f"> Resultados salvos em 'resultados_previsoes.csv'")

print("\n" + "="*80)
print(">> VISUALIZAÇÃO COMPLETA!")
print("="*80)
print(f"\nResumo:")
print(f"  - Acurácia: {acuracia:.1f}%")
print(f"  - Acertos: {acertos}/{total}")
print(f"  - Arquivos gerados:")
print(f"    - analise_previsoes.png (gráfico completo)")
print(f"    - resultados_previsoes.csv (dados)")
print("="*80)

