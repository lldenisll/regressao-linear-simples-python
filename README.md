# Regressão Linear Simples

## Objetivo
Esse repositório tem como objetivo guiar um estudo em machine learning em português, o conteúdo desse repositório foi baseaod no curso de Machine Learning da IBM, no livro An Introduction to Machine Learning (*Miroslav Kubat*) e no livro Introduction to Statistics and Data Analysis (*Christian Heumann and Michael Schomaker Shalabh*). Neste módulo estudaremos a Regressão Linear Simples.

## Regressão Linear Simples, o que é, aonde vive do que se alimenta? :trollface: 
### Regressão
Regressão é o processo de prever um valor contínuo.
Nesse caso, uma variavel independente (geralmente chamada de X) causa algum efeito em uma variável dependente (geralmente chamada de Y).
Ok, ainda parece um bocado confuso, vamos ver alguns exemplos:
- Temperatura na praia (X) VS Taxa de ocupação de hotéis no litoral (Y)
- Altura de uma pessoa (X) VS Peso de uma pessoa (Y)
### O modelo Linear
<img src="./Captura de Tela 2020-07-19 às 15.00.46.png" width = "30%">
A figura acima nos mostra uma associação linear positiva entre X e Y: Quanto maior o valor de X, maior será o valor de Y (e vice-versa), caso esse modelos e referisse ao nosso exemplo da praia, quanto maior a temperatura na praia, maior seria a taxa de ocupação dos hotéis.

### Fórmula e quem é quem nessa bagunça

<img src="./formula.png" width = "20%">

- Y => Nosso amigo Y de chapéu :womans_hat:, é a nossa variável dependente, para encontrarmos o valor dela, dependemos do valor de outra variável que existe independente de Y. No nosso caso, faz calor ou frio na praia (variável X), independente da taxa de ocupacão dos hotéis, mas é inegável que existe uma correlação entre estar calor e os hotéis na praia estarem lotados.

- θ₁ => O têta 1 é o coeficiente angular (Slope ou gradiant), e é representado pela fórmula abaixo:


-  θ₀ =>  O têta zero é o Intercepto (intercept), é representado pela fórmula abaixo:   


- X₁ => 


