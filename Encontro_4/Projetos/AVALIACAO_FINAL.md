# UNIVERSIDADE DE FORTALEZA (UNIFOR)
## Especialização em Engenharia de Dados — Turma 02
## Avaliação Final: Engenharia de Machine Learning em Produção (MLOps Inicial)

**Disciplina:** Machine Learning para Engenharia de Dados  
**Professor:** Cassio Pinheiro  
**Prazos:**
* **Desenvolvimento:** 22/05/2026 (sexta-feira) e 23/05/2026 (sábado)
* **Encontro Síncrono (Presencial):** Sábado (23/05) das 08:00 às 12:00
* **Pitch Final:** Sábado (23/05) a partir das 11:30 (3 minutos por equipe)
* **Envio dos Entregáveis:** Até o final do dia de sábado (23/05/2026) às 23:59

---

## 1. Contexto e Intenção Pedagógica

No contexto da engenharia de dados, desenvolver um modelo preditivo com alta performance não é suficiente se esse modelo não for implantável, reprodutível e robusto frente a novos dados brutos. O objetivo desta avaliação final é colocar o aluno no papel de um **Engenheiro de Machine Learning (ML Engineer)**.

Cada equipe ficará responsável por um dos 6 projetos temáticos disponíveis. O desafio consiste em construir um pipeline de dados unificado com o `scikit-learn` utilizando as melhores práticas do **Jeito Engenheiro**, assegurando a prevenção absoluta de *Data Leakage* (vazamento de dados), otimizando hiperparâmetros, serializando o pipeline completo em un arquivo `.joblib` e implementando um simulador de inferência de produção tolerante a falhas capaz de processar um payload de dados brutos (dicionário Python).

---

## 2. Entregáveis Obrigatórios

Cada equipe deverá entregar um pacote compactado (`.zip`) ou link de repositório contendo os seguintes arquivos:

1. **`notebook.ipynb`**: O Jupyter Notebook do respectivo projeto preenchido, executado do início ao fim e documentado (com gráficos e explicações).
2. **`pipeline_[projeto].joblib`**: O pipeline completo do `scikit-learn` exportado usando a biblioteca `joblib` contendo o pré-processador (`ColumnTransformer`) e o estimador ajustado.
3. **Resumo Executivo (1 a 2 páginas)**: Documento explicativo curto (em formato Markdown `.md` ou PDF) descrevendo a arquitetura do projeto.

### Mínimos Técnicos Exigidos
* Separação de dados de treino e teste feita imediatamente após a carga e antes de qualquer etapa de limpeza ou engenharia analítica.
* Uso obrigatório de `Pipeline` e `ColumnTransformer` para agrupar todas as transformações de dados (imputers de nulos, scalers e encoders de categóricas).
* Implementação obrigatória da função de simulação de produção `predizer_x(payload)` tolerante a valores nulos ou faltantes simulados no payload de produção.

---

## 3. Métricas de Avaliação (Composição da Nota de 0 a 10)

A nota final da avaliação será calculada com base na soma dos desempenhos obtidos em 6 dimensões analíticas fundamentais:

| Dimensão | Peso | Critério de Medição |
| :--- | :---: | :--- |
| **D1. Análise Exploratória e Entendimento do Problema** | 1,5 | Mapeamento inicial do conjunto de dados, verificação de tipos, identificação de desbalanceamento de classes, dados ausentes e anomalias. Formulação de hipóteses analíticas sobre as variáveis explicativas. |
| **D2. Pré-processamento e Feature Engineering com Pipelines** | 2,5 | Criação de transformadores de colunas (`ColumnTransformer`) robustos e aplicação dos imputadores, codificadores e escaladores sem qualquer vestígio de **Data Leakage**. |
| **D3. Modelagem de Machine Learning e Sintonia de Hiperparâmetros** | 2,0 | Experimentos com no mínimo dois algoritmos diferentes (ex: Regressão Linear, Árvore de Decisão, Random Forest ou Gradient Boosting) e busca de parâmetros ótimos via `GridSearchCV` ou `RandomizedSearchCV`. |
| **D4. Validação Fiel e Desempenho Alvo** | 1,5 | Execução da avaliação final apenas sobre o conjunto de teste isolado e alcance do patamar mínimo da **métrica alvo** estipulada para o projeto da equipe. |
| **D5. Serialização e Simulação de Produção (MLOps)** | 1,5 | Exportação limpa do pipeline unificado final usando a biblioteca `joblib` e criação da função de predição robusta capaz de receber dados brutos em dicionário Python e realizar a inferência. |
| **D6. Reprodutibilidade, Organização e Apresentação (Pitch)** | 1,0 | Execução do notebook de cima a baixo sem falhas, fixação de sementes aleatórias (`random_state`) para reprodutibilidade e clareza na defesa técnica do projeto durante o Pitch. |

**Cálculo da Nota Final:**  
$$\text{Nota Final} = D1 + D2 + D3 + D4 + D5 + D6$$

---

### 3.1 Escala de Desempenho por Dimensão

Para cada uma das dimensões, o professor atribuirá a nota seguindo os critérios de aproveitamento a seguir:

| Nível | Faixa de Aproveitamento | Descrição do Desempenho |
| :---: | :---: | :--- |
| **N4 — Excelente** | 90% a 100% | Entrega completa, tecnicamente correta, bem documentada e justificada. |
| **N3 — Bom** | 75% a 89% | Entrega correta com pequenas omissões conceituais ou de escrita. |
| **N2 — Satisfatório** | 60% a 74% | Entrega parcial, com lacunas técnicas ou metodológicas moderadas. |
| **N1 — Insuficiente** | 0% a 59% | Erros conceituais graves, código quebrado ou ausência do entregável. |

---

### 3.2 Indicadores Objetivos por Dimensão (Checklist do Avaliador)

#### D1. Análise Exploratória e Entendimento do Problema (1,5 pt)
- [ ] Apresenta análise do tamanho dos dados, tipos de colunas e estatísticas básicas.
- [ ] Diagnostica o percentual de dados faltantes e a distribuição da variável target (desbalanceamento/distribuição).
- [ ] Formula ao menos duas hipóteses lógicas sobre a relação entre atributos explicativos e o target.

#### D2. Pré-processamento e Feature Engineering com Pipelines (2,5 pts)
- [ ] Isola o conjunto de teste antes de aplicar qualquer transformação (exemplo: `train_test_split`).
- [ ] Aplica transformadores separados para variáveis numéricas e categóricas usando `ColumnTransformer`.
- [ ] Não utiliza funções globais do Pandas (como `.fillna()` ou `.get_dummies()`) no conjunto completo pré-split para evitar Data Leakage.
- [ ] Justifica tecnicamente a escolha de imputadores, escaladores e codificadores de categorias textuais.

#### D3. Modelagem de Machine Learning e Sintonia de Hiperparâmetros (2,0 pts)
- [ ] Avalia o desempenho básico de no mínimo dois algoritmos (ex: Random Forest vs Árvore Simples).
- [ ] Implementa busca de hiperparâmetros (Grid ou Randomized Search) utilizando validação cruzada do próprio scikit-learn acoplada ao pipeline.
- [ ] Garante que os nomes dos parâmetros do estimador na busca estejam referenciados corretamente com o prefixo da etapa do pipeline (ex: `model__n_estimators`).

#### D4. Validação Fiel e Desempenho Alvo (1,5 pt)
- [ ] Reporta métricas de regressão (MAE, RMSE, $R^2$) ou classificação (Recall, Precision, F1-Score, AUC-ROC, Balanced Accuracy) no conjunto de teste.
- [ ] Compara objetivamente os modelos testados por meio de tabela consolidada.
- [ ] Atinge os limiares de métrica alvo estabelecidos para o respectivo projeto da equipe.

#### D5. Serialização e Simulação de Produção (MLOps) (1,5 pt)
- [ ] Serializa o pipeline unificado completo (`preprocessor` + `model`) utilizando `joblib.dump()`.
- [ ] Cria função de teste `predizer_[projeto](payload)` que lê o arquivo `.joblib` em disco.
- [ ] Converte o dicionário bruto recebido (payload de inferência) em um DataFrame Pandas de 1 linha.
- [ ] Testa com sucesso a inferência sobre dados novos que contenham valores nulos simulados sem que a aplicação lance exceções.

#### D6. Reprodutibilidade, Organização e Apresentação (Pitch) (1,0 pt)
- [ ] Garante que o notebook execute do início ao fim utilizando o comando "Restart & Run All".
- [ ] Usa a semente aleatória `random_state` fixa em todas as divisões e modelos estimadores.
- [ ] Apresenta de forma concisa e clara a narrativa técnica do projeto durante o Pitch.

---

## 4. Penalidades Transversais

As penalidades a seguir serão aplicadas diretamente sobre a nota total final (mínimo de 0,0):

> [!WARNING]
> * **-2,0 pontos**: Ocorrência comprovada de **Data Leakage** (aplicar transformações como escalonamento ou imputações na base de dados inteira antes da divisão treino/teste).
> * **-1,0 ponto**: Notebook Jupyter contendo células quebradas que impedem a execução direta ou importação ausente.
> * **-1,0 ponto**: Ausência do arquivo serializado `.joblib` ou falha na execução da função de predição do payload bruto.
> * **-0,5 ponto**: Falta de fixação do `random_state` (o que impede a replicação dos experimentos e das métricas reportadas pela equipe).
> * **-0,5 ponto**: Notebook sem ordem lógica ou sem a documentação das análises exploratórias.

---

## 5. Critérios de Excelência (Bônus Opcional de até +0,5 ponto)

Para equipes que desejarem atingir a nota de destaque, o professor aplicará bônus cumulativo nos seguintes casos:

> [!TIP]
> * **+0,2 ponto**: Implementação de modelos avançados como `XGBoost`, `LightGBM` ou `CatBoost` no pipeline e comparação estruturada com os modelos base do scikit-learn.
> * **+0,2 ponto**: Demonstração de validação cruzada robusta com tratamento de desbalanceamento de classes usando técnicas específicas (ex: SMOTE ajustado estritamente no pipeline de treino usando a biblioteca `imblearn`).
> * **+0,1 ponto**: Clareza executiva, dinamismo e design impecável dos slides de suporte utilizados no Pitch final de 3 minutos.

---

## 6. Dinâmica de Apresentação (Pitch de 3 Minutos)

No final da manhã de sábado (23/05/2026), cada equipe deverá defender o seu projeto de MLOps por meio de um Pitch executivo rápido. A estrutura recomendada é:

1. **O Problema e Dataset (30 segundos)**: Qual era o objetivo e os desafios da base de dados.
2. **Engenharia Analítica (45 segundos)**: Como a equipe desenhou o pré-processador do pipeline de dados.
3. **Modelagem e Otimização (45 segundos)**: Algoritmos testados e parâmetros sintonizados.
4. **Resultados e Métrica Alvo (45 segundos)**: Desempenho final no teste e confirmação do alcance da meta de negócio.
5. **Demonstração da Inferência (15 segundos)**: Prova visual do carregamento do `.joblib` e predição no payload bruto de produção.

---

## 7. Estrutura Recomendada do Notebook

As equipes devem seguir a estrutura definida a seguir dentro do arquivo `.ipynb` do seu projeto:

1. **Passo 1: Importações e Carga dos Dados** (Separação imediata de X e y; divisão treino/teste com semente fixa).
2. **Passo 2: Construção do Pré-processador (`ColumnTransformer`)** (Imputadores, Encoders e Scalers definidos de forma declarativa).
3. **Passo 3: Criação do Pipeline Final com o Modelo** (Acoplamento do pré-processador e escolha do estimador inicial).
4. **Passo 4: Otimização de Hiperparâmetros** (Uso de buscas em grade/aleatória acopladas ao pipeline).
5. **Passo 5: Avaliação do Modelo** (Avaliação fiel sobre X_test e comparação estruturada).
6. **Passo 6: Serialização do Pipeline** (Exportação via `joblib`).
7. **Passo 7: Simulação de Produção** (Função de predição a partir do payload de dicionário carregando o arquivo gerado).

---

## 8. Instruções de Entrega

O envio do entregável final deve ser efetuado até as **23:59 de sábado (23/05/2026)**.
O arquivo compactado contendo o notebook finalizado, o arquivo do modelo serializado `.joblib` e o relatório executivo (Markdown ou PDF) deve ser submetido de acordo com as instruções diretas de envio do professor em sala de aula (Ex: Upload no Classroom, e-mail do professor cassiopo7@gmail.com ou via repositório Git).
