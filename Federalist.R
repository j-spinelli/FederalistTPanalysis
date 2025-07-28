rm(list = ls())

# Se cargan librerias
library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)
library(scales)
library(tsne)
library(LDAvis)
library(wordcloud)


path_archivos <- 
  list.files("corpus", pattern = "\\.txt$", full.names = TRUE)

coleccion <- tibble(texto = character(),
                    documento = character())

# Dado que todos los documentos del corpus comienzan con la frase de target_sentence,
#Se procede a removerla del dataframe para que no afecte al analisis.

target_sentence <- "To the People of the State of New York"

for (path in path_archivos) {
  lineas <- read_lines(path)
  modified_content <- lineas[!grepl(target_sentence, lineas)]
  writeLines(modified_content, path)
  temporal <- tibble(texto = lineas,
                     documento = basename(path))
  coleccion <- bind_rows(coleccion, temporal) 
}

# Se crea el listado de stopwords en ingles
stopwords_en <-
  get_stopwords("en") %>% 
  rename(palabra = word)


# Dado que "United States" es un termino recurrente y con suficiente peso como para considerarlo en el analisis,
# se contruyen dos df con unigramas y bigramas para luego unirlos en un solo df que contenga unigramas y el bigrama "united states"

coleccion_unigrams <-
  coleccion %>%
  unnest_tokens(palabra, texto) %>% 
  anti_join(stopwords_en) %>% 
  filter(!str_detect(palabra, "[0-9]+")) %>% # elimina números
  filter(nchar(palabra) > 1) %>% # elimina palabras como "ó", "á", "é"
  count(documento, palabra, sort = TRUE) # cuenta ocurrencias de cada palabra en su documento


coleccion_bigrams <- coleccion %>%
  unnest_tokens(palabra, texto, token = "ngrams", n = 2) %>% 
  anti_join(stopwords_en) %>% 
  filter(!str_detect(palabra, "[0-9]+")) %>% # elimina números
  filter(nchar(palabra) > 1) %>% # elimina palabras como "ó", "á", "é"
  filter(palabra == "united states")%>%
  count(documento, palabra, sort = TRUE)


coleccion <- bind_rows(coleccion_unigrams, coleccion_bigrams)


coleccion %>% filter(palabra == "united states")

#Se realizará un breve analisis exploratorio de los terminos mas utilizados en el corpus

# Se crea un dataframe que organiza la frecuencia de terminos de forma descendente
top_terms_df <- coleccion %>%
  group_by(palabra) %>%
  summarize(total = sum(n), .groups = 'drop')%>% 
  arrange(desc(total))

# Se crea una wordcloud con los 50 terminos mas utilizados
wordcloud(top_terms_df$palabra, top_terms_df$total, scale=c(3,0.5), min.freq = 0, max.words = 50)  


# Se grafican los 10 terminos mas utilizados para una mejor visualización
top_terms_df %>%
  head(10) %>%
  ggplot(aes(x = reorder(palabra, total), y = total, fill = palabra, label = total)) +
  geom_bar(stat = "identity") +
  geom_text( size = 4, color = "white", position = position_stack(vjust = 0.5)) +  # Add frequency labels on top of bars
  theme_minimal() +
  theme(legend.position = "none") +
  ylab("Número de veces que aparecen") +
  xlab(NULL) +
  ggtitle("Términos más frecuentes") +
  coord_flip()


## TOPIC MODELLING. De aqui en adelante, el proceso es casi identico a lo visto en clase.
  
# Se crea una Document Term Matrix 
coleccion_dtm <-
  coleccion %>% 
  cast_dtm(documento, palabra, n)

# Se corre LDA
coleccion_lda <- LDA(coleccion_dtm, k = 4, control = list(seed = 42))

rm(coleccion_lda)

# diez palabras mas representativas de cada topico
terms(coleccion_lda, 10) %>% as_tibble

#Extraer probabilidades por palabra dentro de cada topico
topicos_coleccion <-
  tidy(coleccion_lda, matrix = "beta") %>% # Esta línea puede requerir instalar el paquete reshape2
  arrange(topic, desc(beta))

topicos_coleccion %>%
  filter(topic == "government")
topicos_coleccion

#Graficar primera diez palabras que definen cada topico
top_terms_model <- topicos_coleccion %>%
  group_by(topic) %>%
  top_n(10, beta) %>% 
  ungroup() %>%
  arrange(topic, -beta)

top_terms_model %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(vars(topic), scales = "free") +
  coord_flip() + 
  ggtitle("Principales palabras de cada tópico")


# distinción de topicos en un histograma // gamma es la probabilidad de cada documento de pertenecer a cierto topico
coleccion_gamma <- tidy(coleccion_lda, matrix = "gamma")

coleccion_gamma %>%
  mutate(gamma = factor(round(gamma))) %>%
  ggplot() +
  aes(x = topic, y = document, fill = gamma) +
  geom_tile() +
  scale_colour_gradient2() + 
  ggtitle("Principales tópicos en cada documento")

#Visualización de datos de forma interactiva
terminos_y_topicos <- posterior(coleccion_lda)
coleccion_dtm <- as.matrix(coleccion_dtm)

svd_tsne <- function(x) tsne(svd(x)$u)
json <- createJSON(
  phi = terminos_y_topicos$terms, 
  theta = terminos_y_topicos$topics, 
  doc.length = rowSums(coleccion_dtm), 
  vocab = colnames(coleccion_dtm), 
  term.frequency = colSums(coleccion_dtm),
  mds.method = svd_tsne,
  plot.opts = list(xlab="", ylab="")
)
serVis(json, out.dir = "LDAvis") # Guarda el resultado en el directorio "LDAvis"
