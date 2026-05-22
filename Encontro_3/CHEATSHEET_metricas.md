# Cheatsheet · Algoritmos e Métricas de Avaliação

**Encontro 3 — Machine Learning para Engenharia de Dados — UNIFOR**
Material de apoio do professor. Profundidade que não cabe no slide.

---

## Parte 1 · Algoritmos de Classificação

### 1.1 Quando escolher cada família

| Família | Use quando | Evite quando | Hiperparâmetros críticos |
|---|---|---|---|
| **Logistic Regression** | Baseline rápido, interpretabilidade obrigatória (crédito, compliance), probabilidades calibradas naturalmente | Não-linearidades fortes, interações entre features | `C` (inverso de regularização), `penalty` (l1/l2/elasticnet), `class_weight='balanced'` |
| **Decision Tree** | Auditoria por regras, prototipagem rápida, dataset pequeno | Necessidade de generalização — overfit fácil | `max_depth`, `min_samples_leaf`, `ccp_alpha` (post-pruning) |
| **Random Forest** | Tabular sem tempo para tuning, robustez out-of-the-box, importância de features | Latência muito baixa (modelo grande), extrapolação fora do treino | `n_estimators`, `max_features`, `max_depth`, `class_weight` |
| **Gradient Boosting (XGBoost / LightGBM / CatBoost)** | SOTA em tabular, Kaggle, decisões críticas com tuning bem feito | Time não tem expertise em hiperparâmetros, decisão precisa ser explicada linha a linha | `learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| **SVM (RBF kernel)** | Alta dimensão (texto, bio), poucos exemplos | N > 50k linhas (escala mal), saída probabilística sem Platt scaling | `C`, `gamma`, `kernel` |
| **KNN** | Sistemas de recomendação simples, dados realmente locais | Alta dimensão (curse of dimensionality), latência baixa | `n_neighbors`, métrica de distância, `weights='distance'` |
| **Naive Bayes** | Classificação de texto (spam, sentimento), baseline ultra-rápido | Features fortemente dependentes (assumption falhou) | `alpha` (Laplace smoothing) |
| **MLP / Redes Neurais** | Não-tabular: imagem, áudio, texto longo, embeddings | Tabular puro com pouca feature engineering (perde para GBM) | arquitetura, `lr`, batch size, regularização (dropout, weight decay) |

### 1.2 Receita prática

1. Comece com `LogisticRegression(class_weight='balanced')`. Métrica baseline registrada.
2. Suba para `RandomForestClassifier(n_estimators=300)`. Sem tuning ainda.
3. Se o salto justifica complexidade, vá para `XGBClassifier` / `LGBMClassifier` com Bayesian Optimization (`optuna`).
4. **Pare cedo.** Se o XGBoost não bate o RF por mais de 1–2 pp na métrica de negócio, escolha o RF — manutenção é mais barata.

---

## Parte 2 · Algoritmos de Regressão

### 2.1 Mapa rápido

| Família | Quando faz sentido | Cuidados |
|---|---|---|
| **OLS (Linear Regression)** | Relações aproximadamente lineares, interpretabilidade | Sensível a multicolinearidade e outliers |
| **Ridge / Lasso / Elastic Net** | Muitas features, risco de overfit | Lasso zera coeficientes — útil para seleção |
| **Decision Tree Regressor** | Não-linearidade simples, captura threshold | Função em degraus — péssimo para extrapolar |
| **Random Forest Regressor** | Tabular geral, baixa engenharia de hiperparâmetros | NÃO extrapola fora do range visto no treino |
| **Gradient Boosting Regressor (XGB / LGBM)** | SOTA em pricing, demanda, forecasting tabular | Calibração do `learning_rate` é crítica |
| **SVR** | Pequenas amostras com kernel | Escala mal para N grande |
| **KNN Regressor** | Interpolação local, séries com sazonalidade simples | Sensível a outliers e dimensionalidade |
| **MLP / LSTM / TCN** | Time-series com sazonalidade complexa, multivariada | Precisa de muito dado + compute |

### 2.2 Forecasting time-series — adendo

Para séries temporais, considere **antes** da família tabular:
- **Naïve / Seasonal Naïve** como baseline obrigatório
- **ETS** (Exponential Smoothing — `statsmodels`)
- **ARIMA / SARIMAX**
- **Prophet** (Meta) — bom para séries com sazonalidade clara
- **NeuralProphet, N-BEATS, Temporal Fusion Transformer** — quando há features exógenas + scale

> O erro mais comum em time-series é **embaralhar o train/test**. Sempre split temporal — nunca aleatório.

---

## Parte 3 · Métricas de Classificação

### 3.1 Matriz de Confusão — derivações completas

```
                     Previu Positivo     Previu Negativo
Real Positivo   |       TP       |       FN
Real Negativo   |       FP       |       TN
```

| Métrica | Fórmula | O que mede | Quando é a métrica certa |
|---|---|---|---|
| Acurácia | (TP+TN)/total | % de acertos | **APENAS** datasets balanceados |
| Precision | TP/(TP+FP) | dos alertados, quantos eram reais | spam, marketing, quando FP é caro |
| Recall (Sensitivity / TPR) | TP/(TP+FN) | dos positivos reais, quantos detectei | saúde, fraude, quando FN é caro |
| Specificity (TNR) | TN/(TN+FP) | dos negativos reais, quantos liberei | controle de FP em screening |
| F1 | 2·P·R/(P+R) | média harmônica P e R | resumo único quando ambos pesam |
| Fβ | (1+β²)·P·R/(β²·P + R) | F1 com peso β em Recall | β=2 para saúde, β=0.5 para spam |
| Balanced Accuracy | (Recall + Specificity) / 2 | accuracy corrigida p/ desbalanceamento | substituto de accuracy em classes raras |
| MCC (Matthews) | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | correlação −1 a +1 | métrica única em multiclasse desbalanceada |
| LogLoss | −(1/n)Σ[y·log(p) + (1−y)·log(1−p)] | qualidade da probabilidade calibrada | quando .predict_proba() vira decisão |

### 3.2 Macro vs Micro vs Weighted (multiclasse)

- **Macro-F1:** média simples dos F1 por classe. Trata classe rara igual à frequente. **Reporte sempre que houver desbalanceamento.**
- **Micro-F1:** soma TP/FP/FN globalmente e calcula. Dominado pelas majoritárias. Em multiclasse, micro-F1 = acurácia.
- **Weighted-F1:** média ponderada pelo support de cada classe. Default razoável para relatório executivo.

### 3.3 AUC-ROC vs AUC-PR — a regra prática

| Critério | AUC-ROC | AUC-PR |
|---|---|---|
| Eixo X | FPR | Recall |
| Eixo Y | TPR (Recall) | Precision |
| Baseline | 0,5 | prevalência da classe positiva |
| Adequado a | classes balanceadas (5–95%) | classes muito desbalanceadas (< 5%) |
| Sensibilidade a FP em N grande | baixa (FPR muda pouco) | alta (Precision cai rápido) |
| Pergunta que responde | "o modelo separa classes?" | "consigo confiar nos positivos previstos?" |

**Resumo:** se a positiva é rara (fraude, anomalia, doença em screening), reporte AUC-PR.

### 3.4 Calibração — métrica esquecida

Modelo pode ter AUC alta mas probabilidades péssimas. Em crédito, marketing personalizado e medicina, **a probabilidade vira decisão financeira** — calibração importa:

- **Brier Score:** `mean((p − y)²)` — quanto menor, melhor a calibração.
- **Reliability diagram:** plote `predict_proba` em decis vs taxa real de positivos.
- Ferramentas: `CalibratedClassifierCV(method='isotonic'|'sigmoid')` para corrigir.

---

## Parte 4 · Métricas de Regressão

### 4.1 Fórmulas e características

| Métrica | Fórmula | Unidade | Sensível a outliers? | Quando usar |
|---|---|---|---|---|
| MAE | (1/n)·Σ\|y−ŷ\| | mesma do target | baixa | KPI conversado com negócio |
| MSE | (1/n)·Σ(y−ŷ)² | quadrado da unidade | alta | **função de perda** de treino |
| RMSE | √MSE | mesma do target | alta | forecast, demanda, LTV |
| MAPE | (1/n)·Σ\|y−ŷ\|/\|y\| | % | alta + explode em y→0 | comparação entre escalas |
| sMAPE | (1/n)·Σ 2\|y−ŷ\|/(\|y\|+\|ŷ\|) | % | média | substituto do MAPE quando y pode ser zero |
| WAPE | Σ\|y−ŷ\| / Σ\|y\| | % | média | demanda agregada por revenue |
| MedAE | mediana(\|y−ŷ\|) | mesma do target | nenhuma | quando outliers são ruído |
| R² | 1 − SSres/SStot | adimensional | sim | comunicação executiva (% explicado) |
| Quantile Loss (Pinball) | parametrizada por τ | mesma do target | depende de τ | forecast com intervalos de confiança |
| MASE | MAE / MAE_naïve | adimensional | baixa | benchmark relativo em time-series |

### 4.2 Exemplo numérico de outlier impactando RMSE

| ID | y real | ŷ predito | erro | erro² |
|---|---|---|---|---|
| 1 | 500.000 | 505.000 | 5.000 | 25.000.000 |
| 2 | 480.000 | 475.000 | 5.000 | 25.000.000 |
| 3 | 520.000 | 525.000 | 5.000 | 25.000.000 |
| ... | ... | ... | 5.000 | 25.000.000 |
| 10 | 510.000 | 505.000 | 5.000 | 25.000.000 |
| 11 | 700.000 | 500.000 | **200.000** | **40.000.000.000** |

- **MAE** = (10·5.000 + 200.000) / 11 ≈ R$ 22,7 k
- **RMSE** = √( (10·25M + 40G) / 11 ) ≈ R$ 60,5 k
- Um único outlier triplicou o RMSE. Em pricing, isso é coerente — esse imóvel é o que destrói o portfólio.

### 4.3 Quando R² é negativo

R² < 0 significa que o modelo prediz **pior do que prever sempre a média do target**. Não é um bug — é diagnóstico.

Causas mais comuns:
1. Treino e teste vêm de distribuições diferentes (data drift).
2. Modelo está overfitting no treino e generalizando péssimo.
3. Features de teste têm NaN não tratado.
4. Cálculo do R² em série temporal sem deseazonalizar — a média do target já é não-estacionária.

---

## Parte 5 · Recipes — escolha rápida em sala

```
TIPO DE PROBLEMA          → MÉTRICA PRIMÁRIA       → MÉTRICA SECUNDÁRIA
─────────────────────────────────────────────────────────────────────
Fraude (0,1%)             → AUC-PR                 → Recall@Precision=0.30
Spam                      → Precision              → F0.5
Diagnóstico médico        → Recall (Sensibilidade) → F2 · Specificity
Aprovação de crédito      → AUC-ROC                → KS, calibração (Brier)
Churn                     → AUC-ROC                → Lift no top 10%
Roteamento (multiclasse)  → Macro-F1               → Matriz de confusão por classe
Multi-label tags          → Hamming Loss           → micro-F1
─────────────────────────────────────────────────────────────────────
ETA (logística)           → MAE                    → P90 do erro
Demanda por SKU           → WAPE / RMSE            → MAPE por faixa de preço
LTV                       → RMSE                   → Calibração por decis
Pricing dinâmico          → MAPE                   → MedAE
Forecast servidor (infra) → Pinball Loss (τ=0.9)   → MAPE
AVM (imóveis)             → MAPE                   → MedAE (5%/50%/95%)
```

---

## Parte 6 · Erros que matam o modelo na prática

1. **Acurácia em desbalanceado.** Sempre cheque `value_counts()` antes de reportar accuracy.
2. **AUC-ROC em fraude.** FPR varia pouco quando negativa é gigante. Use AUC-PR.
3. **Otimizar F1 por inércia.** Decida o erro caro PRIMEIRO; F1 talvez não seja o KPI.
4. **MAPE com y → 0.** Use sMAPE ou MASE.
5. **R² alto em time-series.** Tendência explica boa parte da variância sozinha. Sempre teste contra naïve.
6. **Avaliar no mesmo dataset do tuning.** Holdout final separado, sempre. Use `nested cross-validation` em projetos críticos.
7. **Threshold = 0,5 fixo.** É default da biblioteca, não decisão de negócio.
8. **Reportar uma métrica só.** Sempre traga 2 ou 3 — uma falha esconde outra.

---

## Parte 7 · Snippets prontos (Python · sklearn)

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
    median_absolute_error, fbeta_score,
)
import numpy as np

# === CLASSIFICAÇÃO ===
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print('AUC-ROC :', roc_auc_score(y_test, y_proba))
print('AUC-PR  :', average_precision_score(y_test, y_proba))
print('F2      :', fbeta_score(y_test, y_pred, beta=2))

# threshold customizado (Recall ≥ 0.95)
prec, rec, thr = precision_recall_curve(y_test, y_proba)
idx = np.argmax(rec >= 0.95)
print('threshold para Recall≥0.95:', thr[idx], '| Precision:', prec[idx])

# === REGRESSÃO ===
y_pred = model.predict(X_test)
print('MAE  :', mean_absolute_error(y_test, y_pred))
print('RMSE :', mean_squared_error(y_test, y_pred, squared=False))
print('MAPE :', mean_absolute_percentage_error(y_test, y_pred))
print('R²   :', r2_score(y_test, y_pred))
print('MedAE:', median_absolute_error(y_test, y_pred))
```
