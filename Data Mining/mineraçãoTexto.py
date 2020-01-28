import matplotlib.pyplot as plt
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import string

#CRIA O CORPUS, PRIMEIRO DIZ A PASTA ONDE ESTÁ OS ARQUIVOS E .* PEGA TODOS
corpus = PlaintextCorpusReader('dados', '.*')

#LISTA COM TODOS OS ARQUIVOS ([] PARA ESPECÍFICOS)
arquivos = corpus.fileids()
#TEXTO DE UM ARQUIVO ESPECÍFICO
text = corpus.raw('100.txt')
#TODOS OS TEXTOS
texto = corpus.raw()
##########################
palavras = corpus.words()
len(palavras)
palavras[0:9]

#CRIA A NUVEM DE PALAVRAS 
stops = stopwords.words('english')
cores = ListedColormap(['orange', 'red', 'blue', 'magenta', 'yellow'])
nuvem = WordCloud(background_color = 'white', colormap = cores,
                  stopwords = stops, max_words = 150)
nuvem.generate(texto)

plt.imshow(nuvem)

palavrasmais = [p for p in palavras if p not in stops]
len(palavrasmais)

string.punctuation
palavrasSemPontuacao = [p for p in palavrasmais if p not in string.punctuation]

frequencia = nltk.FreqDist(palavrasSemPontuacao)
mais_comuns = frequencia.most_common(10)



