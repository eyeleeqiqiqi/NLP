##
## Part 2:
## Use pointwise mutual information to compare words in the movie corpora
##
import os.path
import sys
import heapq
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print "Reading file ", f
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence) # append this list as an element to the list of sentences
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print "Error: corpus file ", f, " does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

#--------------------------------------------------------------
# PMI data structure
#--------------------------------------------------------------
class PMI:
    # Given a corpus of sentences, store observations so that PMI can be calculated efficiently
    def __init__(self, corpus):
        self.Px = defaultdict(int)
        self.Pxy = defaultdict(int)
        self.N = len(corpus)
        self.train(corpus)

    def train(self,corpus):
        for i in range(len(corpus)):
            sent = list(set(corpus[i]))
            for j in range(len(sent)):
                self.Px[sent[j]] += 1
                for k in range(j+1,len(sent)):
                    self.Pxy[self.pair(sent[j],sent[k])] += 1
                
    # Return the pointwise mutual information (based on sentence (co-)occurrence frequency) for w1 and w2
    def getPMI(self, w1, w2):
        return float(self.Pxy[self.pair(w1,w2)]*self.N) / (self.Px[w1] * self.Px[w2])

    # Given a frequency cutoff k, return the list of observed words that appear in at least k sentences
    def getVocabulary(self, k):
        wordlist = []
        for word,count in self.Px.iteritems():
            if count >= k:
                wordlist.append(word)
        return wordlist

    # Returns a list of word pairs (2-tuples)
    # Given a list of words, return the pairs of words that have the highest PMI (without repeated pairs, and without duplicate pairs (wi, wj) and (wj, wi))
    def getPairsWithMaximumPMI(self, words, n):
        h = []
        words = set(words)
        for (word1,word2),count in self.Pxy.iteritems():
            if word1 in words and word2 in words:
                heapq.heappush(h, (self.getPMI(word1,word2),(word1,word2)) )
                if len(h) >= 5000: 
                    h = heapq.nlargest(n, h)
                        
        h = heapq.nlargest(n, h)        
        topwordlist = []

        for pmi,words in h:
            topwordlist.append(words)
        return topwordlist

    #-------------------------------------------
    # Provided PMI methods
    #-------------------------------------------
    # Writes the list of wordPairs to a file, along with each pair's PMI
    def writePairsToFile(self, wordPairs, filename): 
        file=open(filename, 'w+')
        for (wi, wj) in wordPairs:
            print >>file, str(self.getPMI(wi, wj))+" "+wi+" "+wj

    # Helper method: given two words w1 and w2, returns the pair of words in sorted order
    # That is: pair(w1, w2) == pair(w2, w1)
    def pair(self, w1, w2):
        return (min(w1, w2), max(w1, w2))

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('train.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("luke", "vader")
    print "PMI of \"luke\" and \"vader\": ", lv_pmi
    numPairs = 100
    k = 200
    for k in 2, 5, 10, 50, 100, 200:
        print "\nRunning for k: " + str(k)
        commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
        wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
        pmi.writePairsToFile(wordPairsWithGreatestPMI, "pairs_minFreq="+str(k)+".txt")
