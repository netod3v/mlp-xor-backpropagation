# MLP XOR — Backpropagation

Implementação do zero (sem frameworks) de uma rede Perceptron Multicamadas (MLP) com Backpropagation, aplicada ao problema XOR e a outras funções lógicas (AND, OR, NAND, NOR). Trabalho complementar ao estudo do neurônio Adaline.

Projeto da disciplina de Introdução à Inteligencia Artificial, ministrada pelo Prof. Fábio Oliveira.

Instituto Federal de Brasília (IFB)

Alunos: Guilherme Souza e Pedro Neto

## Estrutura do projeto

```
mlp_xor/
├── requirements.txt
├── src/
│   ├── mlp.py            # implementação da rede (forward, backward, treino)
│   └── experimentos.py   # roda o experimento principal + experimentos A, B, C e funções lógicas
└── results/               # gráficos gerados (.png)
```

## Passo a passo

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/mlp-xor-backpropagation.git
cd mlp-xor-backpropagation
```

### 2. Criar e ativar um ambiente virtual (opcional, recomendado)

```bash
python3 -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Dependências utilizadas: `numpy` e `matplotlib`.

### 4. Executar os experimentos

```bash
cd src
python experimentos.py
```

Isso vai:

- Treinar a MLP no problema XOR (experimento principal)
- Rodar o Experimento A (variação do número de neurônios na camada oculta)
- Rodar o Experimento B (variação da taxa de aprendizagem)
- Rodar o Experimento C (variação da seed de inicialização dos pesos)
- Treinar a rede para as portas AND, OR, NAND, NOR e XOR
- Salvar todos os gráficos gerados na pasta `results/`

Os resultados (número de épocas, erro final, saídas da rede) são impressos no terminal durante a execução.

### 5. Ver os resultados

Os gráficos ficam salvos em `results/`:

- `xor_espaco.png` — representação do problema XOR no plano cartesiano
- `funcoes_ativacao.png` — funções de ativação implementadas
- `convergencia_principal.png` — convergência do experimento principal
- `exp_A_neuronios.png` — variação do número de neurônios ocultos
- `exp_B_taxa.png` — variação da taxa de aprendizagem
- `exp_C_seeds.png` — variação da inicialização dos pesos
- `exp_funcoes_logicas.png` — comparação entre AND, OR, NAND, NOR e XOR

## Artigo

O artigo completo com a descrição teórica, metodologia e discussão dos resultados está disponível em: `[inserir link/nome do PDF aqui]`
