# Avaliação Final: Projetos de Machine Learning em Produção (MLOps)

> [!IMPORTANT]
> **Atenção:** Os laboratórios práticos nesta pasta foram consolidados como a **Avaliação Final Oficial** da disciplina de *Machine Learning para Engenharia de Dados* (Pós-Graduação UNIFOR).
> Para ler as regras completas de entrega e a rubrica detalhada de avaliação (notas de 0 a 10), consulte o arquivo:
> ### 🔗 [Diretrizes da Avaliação Final (AVALIACAO_FINAL.md)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/AVALIACAO_FINAL.md)

---

## Instruções Gerais
1. Cada equipe ficará responsável por implementar **um dos 6 projetos** detalhados abaixo.
2. Cada equipe deve navegar até a pasta correspondente, abrir o notebook `.ipynb` e completar todas as etapas sinalizadas com `# TODO`.
3. O objetivo técnico é estruturar um pipeline completo no scikit-learn (utilizando `Pipeline` e `ColumnTransformer`), sem cometer *Data Leakage*, ajustando hiperparâmetros para atingir a **métrica alvo mínima** exigida.
4. Ao final, a equipe deve exportar o pipeline treinado em um arquivo `.joblib` e codificar um simulador de inferência robusto capaz de receber e predizer um payload bruto.

---

## Os 6 Projetos e Suas Metas de Performance

### 🏡 [Projeto 1: Precificação de Imóveis (Regressão)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_1_California_Housing/Projeto_1_California_Housing.ipynb)
* **Dataset:** California Housing (com dados numéricos faltantes e colunas categóricas textuais).
* **Métrica Alvo:** RMSE < 70.000 USD ou $R^2$ > 0.70.

### 🔥 [Projeto 2: Risco de Queimadas (Classificação Binária)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_2_Queimadas/Projeto_2_Queimadas.ipynb)
* **Dataset:** Forest Fires (UCI) adaptado para predição binária (ocorrência ou não de incêndio).
* **Métrica Alvo:** F1-Score da classe positiva (incêndio) > 0.65.

### 🩺 [Projeto 3: Diagnóstico de Saúde Pública (Classificação Binária)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_3_Saude_Publica/Projeto_3_Saude_Publica.ipynb)
* **Dataset:** Heart Disease (UCI).
* **Métrica Alvo:** Recall para a classe positiva (presença de doença cardíaca) > 0.80.

### 🔢 [Projeto 4: Classificação de Dígitos por Imagem (Classificação Multiclasse)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_4_Classificacao_Digitos/Projeto_4_Classificacao_Digitos.ipynb)
* **Dataset:** Optical Recognition of Handwritten Digits (scikit-learn).
* **Métrica Alvo:** Acurácia Balanceada (Balanced Accuracy) > 0.90.

### 📞 [Projeto 5: Previsão de Churn em Telecom (Classificação Desbalanceada)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_5_Churn_Telecom/Projeto_5_Churn_Telecom.ipynb)
* **Dataset:** IBM Telco Churn (alto desbalanceamento e nulos implícitos na forma de strings vazias).
* **Métrica Alvo:** AUC-ROC > 0.80.

### 🏥 [Projeto 6: Custos Médicos Individuais (Regressão)](file:///Users/cassiopinheiro/Documents/UNIFOR/Machine%20Learning%20Engenharia%20de%20Dados/Encontro_4/Projetos/Projeto_6_Custos_Medicos/Projeto_6_Custos_Medicos.ipynb)
* **Dataset:** Medical Cost Personal Datasets (múltiplas variáveis categóricas textuais).
* **Métrica Alvo:** $R^2$ > 0.75 ou MAE < 3.000 USD.

---

## Passos Requeridos para a Construção do Pipeline
1. **Split Imediato:** Isolar o conjunto de teste (`train_test_split`) com semente aleatória `42` antes de qualquer imputação ou transformação.
2. **ColumnTransformer:** Agrupar em um único transformador as pipelines específicas de tratamento numérico (escala e mediana) e categórico (one-hot encoding e moda).
3. **Pipeline Acoplado:** Montar o pipeline final que executa o `ColumnTransformer` e em seguida o estimador (`model`).
4. **Busca de Hiperparâmetros:** Utilizar `GridSearchCV` ou `RandomizedSearchCV` aplicados diretamente no pipeline para otimizar os parâmetros.
5. **Serialização:** Salvar o pipeline final treinado em formato `.joblib`.
6. **Simulador de Produção:** Codificar e testar a função de predição alimentada por payload bruto que simula requisições reais de uma API.
