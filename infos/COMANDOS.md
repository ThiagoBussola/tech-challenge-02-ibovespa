# 🚀 Comandos Rápidos - Projeto Ibovespa

## 📋 Índice

- [Setup Inicial](#setup-inicial)
- [Treinar Modelos](#treinar-modelos)
- [Visualizar Resultados](#visualizar-resultados)
- [Estrutura de Arquivos](#estrutura-de-arquivos)

---

## ⚙️ Setup Inicial

### 1️⃣ Primeira vez (Instalar dependências):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install libomp  # Necessário para XGBoost (macOS)
```

### 2️⃣ Próximas vezes (Ativar ambiente):

```bash
source venv/bin/activate
```

---

## 🤖 Treinar Modelos

# Treinar modelo

python3 ibovespa_final.py
python3 ibovespa_ATINGIR_75.py

# Ver relatório

python3 gerar_relatorio.py

# Gerar visualizações

python3 visualizar_previsoes.py

**Resultado esperado**: 80% de acurácia ✅  
**Arquivo gerado**: `modelo_final.pkl`
