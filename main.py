import io
from numpy import dot
from numpy.linalg import norm
from scipy.stats import spearmanr



vect_file = "crawl-300d-2M-subword.vec"
wsim_rel_file = "wordsim_relatedness_goldstandard.txt"
wsim_sim_file = "wordsim_similarity_goldstandard.txt"


alphabet = {} #set of unique words in both files with their vectors extracted from 

#fills alphabet dict
#returns matrix of pairs extracted from text file 
# each array in matrix contains 3 elements
# word1 word2 relatedness_value
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
    for key in alphabet:
        alphabet[key] = list(alphabet[key])

#calculates cos of the angle between 2 float vectors
def calc_cos(vec1,vec2):
    scal_prod = dot(vec1,vec2)
    return scal_prod/(norm(vec1)*norm(vec2))


#calculates cosine relatedness for all pairs in the array, and adds calculated result to the array 
def add_fasttext_relatedness(pair_matr):
    for i in range(len(pair_matr)):
        vec1 = alphabet[pair_matr[i][0]]
        vec2 = alphabet[pair_matr[i][1]]
        relatedness = calc_cos(vec1,vec2)
        pair_matr[i].append(relatedness)


# each array describing a pair contains 4 elems
# word1 word2 relatedness from the file, cosine relatedness from wordtext
def calc_spearman(pair_matr):
    wordsim_rel_arr = list(map(lambda elem:elem[2],pair_matr))
    fasttext_rel_arr = list(map(lambda elem:elem[3],pair_matr))
    result,p_val = spearmanr(wordsim_rel_arr,fasttext_rel_arr)
    return result

wsim_rel_pairs = load_word_pairs(wsim_rel_file)
wsim_sim_pairs = load_word_pairs(wsim_sim_file)


load_vectors(vect_file)

add_fasttext_relatedness(wsim_rel_pairs) 
add_fasttext_relatedness(wsim_sim_pairs)

result1 = calc_spearman(wsim_rel_pairs) #spearman correlation betw rel values from wsim_relatedness and cosine rel values 
result2 = calc_spearman(wsim_sim_pairs) #spearman correlation betw rel values from wsim_similarity and ..

print("spearman correlation betw rel values from wsim_relatedness and cosine rel values : {}".format(result1))
print("spearman correlation betw rel values from wsim_similarity and cosine rel values : {}".format(result2))

#output of results into text file
with open("results_relatedness.txt","w") as outfile:
    for elem in wsim_rel_pairs:
        outfile.write(elem[0]+" "+elem[1]+" "+"{:.3f}".format(elem[3])+"\n")

with open("results_similarity.txt","w") as outfile:
    for elem in wsim_sim_pairs:
        outfile.write(elem[0]+" "+elem[1]+" "+"{:.3f}".format(elem[3])+"\n")