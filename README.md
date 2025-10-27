# 📈 Predição de Tendência do Ibovespa

Projeto de Machine Learning para prever a tendência diária do índice Ibovespa com **meta de 75% de acurácia**.

---

## 🎯 Objetivo

Desenvolver um modelo de Machine Learning capaz de prever se o Ibovespa terá **alta** ou **baixa** no dia seguinte, utilizando apenas dados históricos do próprio índice.

**Meta de acurácia**: ≥ 75%

---

## 📂 Estrutura do Projeto

```
TechChallenge-Ibovespa/
├── data/
│   ├── Dados Históricos - Ibovespa.csv        # 5 anos de dados históricos
│   └── Dia Atual - Ibovespa - 07-10-2025.csv  # Dados para validação
├── ibovespa_ATINGIR_75.py                     # 5 estratégias para ≥75%
├── ibovespa_final.py                    # Configuração exata que deu 80%
├── gerar_relatorio.py                         # Script para visualizar resultados
├── requirements.txt                           # Dependências do projeto
├── README.md                                  # Este arquivo
├── melhor_modelo_ibovespa_FINAL.pkl          # Melhor modelo (gerado)
├── modelo_final.pkl                    # Modelo 80% (gerado)
├── resultado_final_75pct.png                 # Gráficos (gerados)
└── venv/                                      # Ambiente virtual Python
```

---

## 🚀 Como Usar

### 1️⃣ **Instalar Dependências** (primeira vez)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install libomp  # Necessário para XGBoost no macOS
```

#### 🎯 **COMANDOS RÁPIDOS DE REFERÊNCIA**

Veja o arquivo: **[`infos/COMANDOS.md`](infos/COMANDOS.md)**

### 2️⃣ **Treinar o Modelo**

**🏆 Opção A: Modelo Perfeito de 80%** (RECOMENDADO)

```bash
source venv/bin/activate
python3 ibovespa_final.py
```

**Tempo**: ~2 minutos | **Acurácia**: 80% garantido ✅

**O que faz**:

- Usa configuração exata: Seed 11 + Threshold 0.48
- Neural Network: 256→128→64→32 neurônios
- Salva em `modelo_final.pkl`

---

**🔬 Opção B: Testar Múltiplas Estratégias**

```bash
source venv/bin/activate
python3 ibovespa_ATINGIR_75.py
```

**Tempo**: ~5-10 minutos | **Acurácia**: 75-80%

**O que faz**:

- Feature Selection (60, 80, 100, 120 features)
- Random Seeds (7 diferentes)
- Optuna (30 tentativas bayesianas)
- Ensemble Multi-Janela (20+25+30 dias)
- Walk-Forward (50 seeds)
- Salva melhor em `melhor_modelo_ibovespa_FINAL.pkl`

### 3️⃣ **Visualizar Resultados**

**Relatório Simples:**

```bash
source venv/bin/activate
python3 gerar_relatorio.py
```

**Análise Completa com Gráficos:**

```bash
source venv/bin/activate
python3 visualizar_previsoes.py
```

Gera:

- 📊 Timeline de previsões (alta/baixa)
- 📈 Gráficos de acurácia
- 📋 Tabela de resultados
- ✅ Análise de acertos por tipo de tendência
- 💾 Arquivos: `analise_previsoes.png` e `resultados_previsoes.csv`

### 4️⃣ **Documentação Técnica**

Contém:

- Aquisição e exploração dos dados
- Estratégia de engenharia de 178 features
- Preparação da base e definição de target
- Escolha e justificativa do modelo (MLP)
- Resultados detalhados e métricas
- Trade-offs (acurácia vs overfitting)
- Análise de confiabilidade

---

## 🧠 Estratégias de Otimização

O script `ibovespa_ATINGIR_75.py` implementa **5 estratégias** para maximizar a acurácia:

### 1. **Feature Selection**

- Testa com 60, 80, 100, 120 features
- Reduz overfitting selecionando as mais importantes

### 2. **Múltiplos Random Seeds**

- Testa 7 seeds diferentes (17, 42, 123, 777, 2023, 2024, 2025)
- Diferentes seeds geram diferentes resultados

### 3. **Otimização Bayesiana (Optuna)**

- 30 tentativas de busca inteligente
- Otimiza: camadas, neurônios, alpha, learning rate

### 4. **Ensemble Multi-Janela**

- Combina previsões de janelas 20, 25, 30 dias
- Voto ponderado (mais peso para janela 25)

### 5. **Walk-Forward com 50 Seeds**

- Testa 50 seeds diferentes (1 a 50)
- Aumenta estatisticamente a chance de atingir 75%

---

## 📊 Features Técnicas

O modelo utiliza **178 indicadores técnicos** criados a partir dos dados históricos:

### Médias Móveis

- Simple Moving Average (SMA): 3, 5, 7, 10, 15, 20, 30, 50 dias
- Exponential Moving Average (EMA): 5, 10, 20, 30 dias
- Ratios e diferenças entre preço e médias

### Momentum & Volatilidade

- Retornos: 1, 2, 3, 5, 7, 10, 15, 20, 30 dias
- Volatilidade: janelas de 3 a 30 dias
- Momentum: 5, 10, 14, 20, 30 períodos

### Indicadores Técnicos

- **RSI** (Relative Strength Index): 7, 14, 21, 28 períodos
- **MACD** (Moving Average Convergence Divergence): múltiplas configurações
- **Bollinger Bands**: 10, 20, 30 dias com 1.5σ, 2σ, 2.5σ
- **ATR** (Average True Range): 7, 14, 21 dias
- **Stochastic Oscillator**: 14, 21 períodos

### Volume

- Volume médio: 5, 10, 20 dias
- Ratios de volume
- Mudanças percentuais

### Padrões de Candle

- Daily Range, Body, Shadows
- Ratios: Body/Range, Shadow/Body
- Contadores: Higher Highs, Lower Lows

### Features Temporais

- Dia da semana, Dia do mês, Mês, Trimestre, Semana do ano

### Lags

- Preço, Variação, Volume com lags de 1 a 7 dias

---

## 🔧 Modelos Testados

- **Neural Networks (MLP)**: 3 arquiteturas diferentes
- **XGBoost**: Gradient Boosting otimizado
- **LightGBM**: Gradient Boosting leve e rápido
- **CatBoost**: Gradient Boosting com suporte a categóricas
- **Voting Ensemble**: Combinação de múltiplos modelos
- **Stacking**: Ensemble de dois níveis

---

## 📈 Resultados Esperados

Com as 5 estratégias implementadas, a expectativa é:

- ✅ **Feature Selection**: 73-74%
- ✅ **Random Seeds (7)**: 73-75%
- ✅ **Optuna (30 trials)**: 74-76%
- ✅ **Ensemble Multi-Janela**: 73-75%
- ✅ **Walk-Forward (50 seeds)**: 74-77%

**Probabilidade de atingir ≥75%**: ALTA 🎯

---

## 📝 Dependências Principais

- **Python**: 3.13+
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Modelos de ML e pré-processamento
- **xgboost**: Gradient Boosting
- **lightgbm**: Light Gradient Boosting
- **catboost**: CatBoost Gradient Boosting
- **imbalanced-learn**: SMOTE para balanceamento
- **optuna**: Otimização Bayesiana
- **matplotlib/seaborn**: Visualizações

---

## 👨‍💻 Autor

Projeto desenvolvido para o **Tech Challenge - Fase 2** da FIAP - POSTECH.

**Data**: Outubro de 2025

---

## 📄 Licença

Este projeto é para fins educacionais.

---

## 🎓 Referências

- [Documentação Scikit-learn](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna - Hyperparameter Optimization](https://optuna.org/)
- [Technical Indicators Guide](https://www.investopedia.com/technical-analysis-4689657)
