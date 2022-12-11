
# ------------------------------------------------
# FEAUSP
# EAD737 - Topicos Avancados de Financas
# Prof. Leandro Maciel (leandromaciel@usp.br)
# ------------------------------------------------

# Script Aula 11 - Redes Neurais Artificiais

# ------------------------------------------------

install.packages("neuralnet")


# Carregar os pacotes necessarios:

library(readxl)
library(neuralnet)
set.seed(450)
# ------------------------------------------------

cat("\f") # Limpar o console

rm(list = ls()) # Limpar todas as variaveis

# ------------------------------------------------

# Carregar os dados IPCA (variacao % mensal Jan/00 a Set/20):

setwd("") #Defina aqui o caminho para o seu diretório onde está salvo o seu arquivo de dados.
IPCA = read_excel("DadosAula11.xlsx")

# Plotar os dados:

plot(IPCA$Data,IPCA$IPCA,type = "l",xlab = "",ylab = "Variacao (%)",main = "IPCA Mensal",col = "blue")

# ------------------------------------------------

# Passo 1: normalizacao dos dados

# Construir função para normalizar (min-max):

normalize = function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# Normalizar os dados:

IPCA_norm = normalize(IPCA$IPCA)

plot(IPCA$Data,IPCA_norm,type = "l",xlab = "",ylab = "Variacao (%)",main = "IPCA Mensal Normalizado",col = "blue")

# ------------------------------------------------

# Passo 2: definir modelo de previsao...

# Quantos valores passados para prever o IPCA do proximo mes?

no = length(IPCA_norm)

base = as.data.frame(cbind(IPCA_norm[1:(no-2)],IPCA_norm[2:(no-1)],IPCA_norm[3:no]))

# ------------------------------------------------

# Passo 3: dividir amostras em treinamento e validacao:

n = 18 # numero de obs deixadas para previsão

IPCAin = base[1:(nrow(base)-n),]
IPCAout = base[(nrow(base)-n+1):(nrow(base)),]

# ------------------------------------------------

# Passo 4: definir estrutura e treinar RNA:

#softplus <- function(x) log(1 + exp(x))

for (i in 1:5) {
  for (j in 1:5) {
    cat(i, " \t\t\t")
    cat(j, " \t\t\t")
  
    softplus <- function(x) cos(x)
    
    modeloRede = neuralnet(
      V3 ~ V1 + V2, # modelo de previsao considerado;
      data = IPCAin, # base de dados treinamento;
      act.fct = softplus, # tangente hiperbolica;
      hidden = c(4,i,j,1) # camadas e neuronios em cada uma.
    )
    
    # Visualizar rede neural:
    
    #plot(modeloRede)
    
    # Guardar a saida do modelo no treinamento:
    
    saida_Treino = as.matrix(modeloRede[["net.result"]][[1]])
    
    # Visualizar ajuste da rede no treinamento:
    
    #plot(IPCA$Data[1:(nrow(base)-n)],IPCAin[,3],xlab = "",ylab = "Variacao (%)",type="l",main = "IPCA Real e Previsto - Amostra Treino")
    #lines(IPCA$Data[1:(nrow(base)-n)],saida_Treino,col="red")
    
    # ------------------------------------------------
    
    # Passo 5: realizar previsões na amostra teste.
    
    previsao = predict(modeloRede,IPCAout[,1:2])
    
    # Visualizacao:
    
    #plot(IPCA$Data[(nrow(base)-n+1):(nrow(base))],IPCAout[,3],xlab = "",ylab = "Variacao (%)",type="l",main = "IPCA Real e Previsto - Amostra Teste")
    #lines(IPCA$Data[(nrow(base)-n+1):(nrow(base))],previsao,col = "red")
    
    
    
    # ------------------------------------------------
    
    
    diff = IPCAout[,3] - previsao
    
    diff_abs = abs(diff)
    #diff_abs
    
    error = sum(diff_abs)
    cat(error, "\n")

  }
}
