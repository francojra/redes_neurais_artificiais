
# Redes Neurais Artificiais ----------------------------------------------------------------------------------------------------------------
# Autora do script: Jeanne Franco ----------------------------------------------------------------------------------------------------------
# Data: 29/09/23 ---------------------------------------------------------------------------------------------------------------------------
# Referência: https://acervolima.com/construindo-uma-rede-neural-simples-na-programacao-r/ -------------------------------------------------

# Introdução -------------------------------------------------------------------------------------------------------------------------------

### Na referência da inteligência artificial, as redes neurais são um conjunto de 
### algoritmos projetados para reconhecer um padrão como o do cérebro humano. 

### Eles interpretam os dados sensoriais por meio de uma espécie de percepção da 
### máquina, rotulagem ou agregação de dados brutos.

### O reconhecimento é numérico, que é armazenado em vetores, nos quais todos os 
### dados do mundo real, sejam imagens, sons, textos ou séries temporais, devem ser 
### traduzidos.

### Uma rede neural pode ser retratada como um sistema que consiste em uma série de 
### nós altamente interconectados, chamados de 'neurônios', que são organizados em 
### camadas que processam informações usando respostas de estado dinâmico a entradas 
### externas.

### Perceptrons são um tipo de neurônios artificiais desenvolvidos nas décadas de 1950 
### e 1960 pelo cientista Frank Rosenbalt, inspirado em trabalhos anteriores de Warren 
### McCulloch e Walter Pitts. Então, como funciona o perceptron? Um perceptron pega 
### várias saídas binárias x 1 , x 2 ,…., E produz uma única saída binária.
### Ele pode ter mais ou menos entradas. Para calcular / computar, os pesos de saída 
### desempenham um papel importante. A saída do neurônio (o ou 1) depende totalmente 
### de um valor limite. O perceptron é que é um dispositivo que toma decisões avaliando 
### as evidências. Variando os pesos e o limite, podemos obter diferentes modelos de 
### tomada de decisão.

### Tenta emular matematicamente o funcionamento do cérebro humano.
### Procura de maneira não linear padrões mais complexos dos dados nas
### variáveis de entrada.
### As redes neurais incluem os dados de entrada, as camadas intermediárias
### que estão ligadas as camadas de entrada através do cálculo da função
### de ativação (função que fornece a resposta da rede) que por sua vez
### está ligada a função de saída.
### A função de saída fornece a resposta da rede neural.
### As redes neurais utilizam vários modelos complexos para melhor tentar
### predizer a resposta.

# Preparando os dados  ---------------------------------------------------------------------------------------------------------------------

### Aqui, vamos usar os conjuntos de dados binários. O objetivo é prever se um 
### candidato será admitido em uma universidade com variáveis como gre, gpa e 
### rank. 

getwd()
data <- read.csv("binary.csv")
str(data)
View(data)

### Dimensionamento dos dados

hist(data$gre)
hist(data$gpa)
hist(data$rank)

### Normalizar os dados do gre

normalize <- function(x) {
        return((x - min(x)) / (max(x) - min(x)))
}

data$gre <- (data$gre - min(data$gre)) / (max(data$gre) - min(data$gre))
hist(data$gre)

data$gpa <- (data$gpa - min(data$gpa)) / (max(data$gpa) - min(data$gpa))
hist(data$gpa)

data$rank <- (data$rank - min(data$rank)) / (max(data$rank) - min(data$rank))
hist(data$rank)

### Pode-se ver a partir da representação dos histogramas acima que gpa, gre e 
### classificação (rank) são escalados no intervalo de 0 a 1. Os dados escalados 
### são usados para se ajustar à rede neural.

# Amostragem dos dados em treinamento e teste ----------------------------------------------------------------------------------------------

### Agora divida os dados em um conjunto de treinamento e um conjunto de teste. 
### O conjunto de treinamento é usado para encontrar a relação entre as variáveis 
### dependentes e independentes enquanto o conjunto de teste analisa o 
### desempenho do modelo. 

### A atribuição dos dados ao conjunto de treinamento e teste é feita por meio de 
### amostragem aleatória. Realizamos amostragem aleatória em R usando a sample()
### função. Use set.seed() para gerar a mesma amostra aleatória sempre e manter 
### a consistência. 

set.seed(222)
inp <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training_data <- data[inp == 1, ]
test_data <- data[inp == 2, ]

View(training_data)
View(test_data)
