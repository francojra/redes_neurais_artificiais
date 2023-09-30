
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

# Instalando uma rede neural ---------------------------------------------------------------------------------------------------------------

### A função neuralnet() nos ajuda a estabelecer uma rede neural para nossos dados. 

### Parâmetros da função neuralnet()

### Fórmula: uma descrição simbólica do modelo a ser ajustado;
### data: um quadro de dados contendo as variáveis especificadas na fórmula;
### hidden: um vetor de inteiros especificando o número de neurônios ocultos 
### (vértices) em cada camada;
### err.fct: 	uma função diferenciável que é usada para o cálculo do erro. 
### Alternativamente, as strings 'sse' e 'ce' que representam a soma dos erros 
### quadráticos e a entropia cruzada podem ser usada;
### linear.output: Se act.fct não deve ser aplicado aos neurônios de saída, 
### defina a saída linear como TRUE, caso contrário, como FALS;
### lifesign: uma string que especifica o quanto a função imprimirá durante 
### o cálculo da rede neural. 'nenhum', 'mínimo' ou 'completo';
### rep: o número de repetições para o treinamento da rede neural;
### algorithm: 	uma string contendo o tipo de algoritmo para calcular a rede 
### neural. Os seguintes tipos são possíveis: 'backprop', 'rprop +', 'rprop-', 
### 'sag' ou 'slr'. 'backprop' refere-se à retropropagação, 'rprop +' 
### e 'rprop-' referem-se à retropropagação resiliente com e sem backtracking 
### de peso, enquanto 'sag' e 'slr' induzem o uso do algoritmo globalmente 
### convergente modificado;
### stepmax: 	os passos máximos para o treinamento da rede neural. Atingir esse 
### máximo leva à interrupção do processo de treinamento da rede neural.

library(neuralnet)

set.seed(333)
n <- neuralnet(admit~gre + gpa + rank,
               data = training_data,
               hidden = 5,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 2,
               algorithm = "rprop+",
               stepmax = 100000)

### A partir da saída acima, concluímos que ambas as repetições convergem. 
### Mas usaremos o orientado por saída na primeira repetição porque dá menos erro 
### (139,80883) do que o erro (147,41304) derivado da segunda repetição. Agora, 
### vamos plotar nossa rede neural e visualizar a rede neural computada. 

plot(n, rep = 1)

### O modelo possui 5 neurônios em sua camada oculta. As linhas pretas mostram 
### as conexões com pesos. Os pesos são calculados usando o algoritmo de 
### retropropagação. 

### A linha azul exibe o termo de polarização (constante em uma equação de 
### regressão).

### Agora gere o erro do modelo da rede neural, junto com os pesos entre 
### as entradas, camadas ocultas e saídas:

n$result.matrix

# Predição ---------------------------------------------------------------------------------------------------------------------------------

### Vamos prever a classificação usando o modelo de rede neural.

output <- compute(n, rep = 1, training_data[, -1])
head(output$net.result)

head(training_data[1, ]) # primeiro dado de previsão
head(training_data[5, ])

# Array de confusão e erro de classificação ------------------------------------------------------------------------------------------------

### Em seguida, arredondamos nossos resultados usando o método  compute() e criamos 
### uma array de confusão para comparar o número de verdadeiros / falsos positivos 
### e negativos. Vamos formar uma array de confusão com dados de treinamento:

output <- compute(n, rep = 1, training_data[, -1])
p1 <- output$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)
tab1 <- table(pred1, training_data$admit)
tab1

### O modelo gera 177 verdadeiros negativos (0's), 34 verdadeiros positivos (1's),
### enquanto há 12 falsos negativos e 58 falsos positivos. Agora, vamos calcular o 
### erro de classificação incorreta (para dados de treinamento).

### {1 - erro de classificação}

1 - sum(diag(tab1)) / sum(tab1)

### O erro de classificação errada chega a ser 24,9%. Podemos aumentar ainda mais 
### a precisão e a eficiência do nosso modelo aumentando a diminuição dos nós e o 
### viés nas camadas ocultas.

### A força dos algoritmos de aprendizado de máquina está em sua capacidade de 
### aprender e melhorar sempre na previsão de uma saída. No contexto das redes 
### neurais, isso implica que os pesos e vieses que definem a conexão entre os 
### neurônios se tornam mais precisos. 