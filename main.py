import io
import fasttext


vect_file = "crawl-300d-2M.vec"
wsim_rel_file = "wordsim_relatedness_goldstandard.txt"
wsim_sim_file = "wordsim_similarity_goldstandard.txt"


alphabet = {} #set of unique words in both files with their vectors extracted from 


#loads word pairs and forms list of unique words
def load_word_pairs(fname):
    global alphabet
    pairs = [] 
    with open(fname,"r") as fin:
        for line in fin:
            w1,w2,relatedness = line.split()
            pairs.append([w1,w2,float(relatedness)])
            if (not(w1) in alphabet):
                alphabet.update({w1:0})
            if (not(w2) in alphabet):
                alphabet.update({w2:0})

    return pairs

def load_vectors(fname):
    global alphabet
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        word = line.split(' ',maxsplit = 1)[0]
        if (alphabet.get(word)!=None):
            tokens = line.rstrip().split(' ') #word + vector float values
            alphabet[word] = map(float, tokens[1:])

wsim_rel_pairs = load_word_pairs(wsim_rel_file)
wsim_sim_pairs = load_word_pairs(wsim_sim_file)


load_vectors(vect_file)

for key in alphabet:
    alphabet[key] = list(alphabet[key])

print(alphabet["tiger"])

