# Sistema-de-Recomenda-o-de-Filmes-com-KNN
Recomendação de filme usando algoritmo KNN
# 🎬 Sistema de Recomendação de Filmes com KNN

## Descrição

Este projeto implementa um **sistema de recomendação de filmes** baseado em **filtragem colaborativa item-item**, utilizando o algoritmo **K-Nearest Neighbors (KNN)** com **distância cosseno**.

A recomendação é feita com base no comportamento dos usuários: filmes são considerados semelhantes quando recebem avaliações parecidas pelos mesmos usuários.

---

## Objetivo

Construir um modelo simples e eficiente capaz de:

* identificar filmes similares;
* recomendar conteúdos com base em um filme de referência;
* demonstrar, na prática, conceitos de Machine Learning aplicados a sistemas de recomendação.

---

## Conceitos Aplicados

* Filtragem colaborativa (Collaborative Filtering)
* Matriz usuário-item
* Similaridade entre vetores
* Distância cosseno
* K-Nearest Neighbors (KNN)
* Matrizes esparsas (Sparse Matrix)

---

##  Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* SciPy
* Scikit-learn

---

## Estrutura do Projeto

```
projeto-recomendacao-filmes/
│
├── movies_metadata.csv
├── ratings.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

O projeto utiliza dados de filmes e avaliações contendo:

* ID do filme
* Título
* Idioma
* Número de avaliações
* Avaliações dos usuários

---

## Etapas do Projeto

1. **Carregamento dos dados**
2. **Pré-processamento**

   * remoção de valores nulos
   * filtragem de usuários ativos
   * seleção de filmes populares
3. **Junção dos dados**
4. **Criação da matriz usuário-item**
5. **Transformação em matriz esparsa**
6. **Treinamento do modelo KNN**
7. **Geração de recomendações**

---

## Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o projeto

```bash
python main.py
```

---

## 💡 Exemplo de Saída

```
Recomendações para: Toy Story

Toy Story 2 | Distância: 0.12 | Similaridade: 0.88
A Bug's Life | Distância: 0.15 | Similaridade: 0.85
Monsters, Inc. | Distância: 0.18 | Similaridade: 0.82
Finding Nemo | Distância: 0.20 | Similaridade: 0.80
Aladdin | Distância: 0.22 | Similaridade: 0.78
```

---

## Interpretação dos Resultados

* **Distância baixa → filmes mais semelhantes**
* **Similaridade alta → maior proximidade entre padrões de avaliação**

A similaridade é calculada a partir da fórmula:

[
\text{similaridade} = 1 - \text{distância}
]

---

## Limitações

* Busca por título exige correspondência exata
* Possíveis filmes com títulos duplicados
* Não trata o problema de *cold start* (novos usuários ou filmes)
* Considera apenas padrões de avaliação (não usa conteúdo do filme)

---

## Possiveis Melhorias Futuras

* Implementar busca aproximada de títulos
* Criar interface com Streamlit
* Comparar com outros algoritmos (SVD, Matrix Factorization)
* Implementar avaliação do modelo
* Adicionar recomendação baseada em conteúdo

---

## Autor

Gabriel Machado

---
