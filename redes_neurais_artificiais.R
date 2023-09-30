
# Redes Neurais Artificiais ----------------------------------------------------------------------------------------------------------------
# Autora do script: Jeanne Franco ----------------------------------------------------------------------------------------------------------
# Data: 29/09/23 ---------------------------------------------------------------------------------------------------------------------------
# Referência: https://www.youtube.com/watch?v=K7y3fF7WoHw ----------------------------------------------------------------------------------

# Introdução -------------------------------------------------------------------------------------------------------------------------------

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

# Análises ---------------------------------------------------------------------------------------------------------------------------------

### Preparando os dados 

### Aqui, vamos usar os conjuntos de dados binários. O objetivo é prever se um 
### candidato será admitido em uma universidade com variáveis como gre, gpa e 
### rank. 

getwd()
data <- read.csv("binary.csv")
str(data)
View(data)



