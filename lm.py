##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
import math
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

# Preprocess the corpus to help avoid sparsity issues
def preprocess(corpus):
    print "Task 0: edit the preprocess function to replace rare words with UNK and add sentence markers"
    tokens = {}
    rare_words = {}
    for i in range(len(corpus)):
        corpus[i] = [start]+corpus[i]+[end]
        for j in range(1,len(corpus[i])-1):
            if(corpus[i][j] in tokens):
                tokens[corpus[i][j]] += 1
                rare_words.pop(corpus[i][j],None)
            else:
                tokens[corpus[i][j]] = 1
                rare_words[corpus[i][j]] = (i,j)
    for word,(sent,index) in rare_words.iteritems():
        corpus[sent][index] = UNK
        tokens.pop(word)
    tokens[UNK]=len(rare_words)
    print "\nNumber of tokens = " + str(len(tokens))
    print "\nNumber of rare words = " + str(len(rare_words))
    for sent in corpus:
        sent = [start]+sent+[end]
    return corpus

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print """Your task is to implement three kinds of n-gram language models:  
      a) an (unsmoothed) unigram model (UnigramModel)
      b) an unsmoothed bigram model (BigramModel)
      c) a bigram model smoothed using absolute discounting (SmoothedBigramModel)"""

    # Generate a sentence by drawing words according to the model's probability distribution
    # Note: think about how to set the length of the sentence in a principled way
    def generateSentence(self):
        print "Implement the generateSentence method in each subclass"
        return "mary had a little lamb ."

    # Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        print "Implement the getSentenceProbability method in each subclass"
        return 0.0

    # Given a corpus, calculate and return its perplexity (normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print "Implement the getCorpusPerplexity method"
        return 0.0

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        file=open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            #print "\n"+sen+str(len(sen.split()))
            prob = self.getSentenceProbability(sen)
            print >>file, prob, " ", sen

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        #print "Subtask: implement the unsmoothed unigram language model"
        self.dist = UnigramDist(corpus)

    def generateSentence(self):
        #print "Implementing generateSentence method for Unigram"
        sent = [start]
        while(1):
            token = self.dist.draw()
            if token == start:
                continue
            if token != end:
                sent.append(token)
            else:  
                break
        sent.append(end)
        return " ".join(sent)

    # Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        #print "Implement the getSentenceProbability method in each subclass"
        p = 0.0
        sent = sen.split()
        for word in range(1,len(sent)):
            p += self.dist.prob(sent[word])
        return p

    def getCorpusPerplexity(self, corpus):
        logp = []
        N = 0
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                if self.dist.counts[word]/self.dist.total == 0:
                    logp.append(0.0)
                else:
                    logp.append(math.log(self.dist.counts[word]/self.dist.total))
                N += 1
        sump = sum(logp) / N
        p = pow(math.e, -sump)
        return p

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        #print "Subtask: implement the unsmoothed bigram language model"
        self.dist =  BigramDist(corpus)
    
    def generateSentence(self):
        #print "Implementing generateSentence method for Bigram"
        sent = [start]
        wordi = start
        while(1):
            token = self.dist.draw(wordi)
            if token != end:
                sent.append(token)
                wordi = token
            else:  
                break
        sent.append(end)
        return " ".join(sent)

    # Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        #print "Implementing the getSentenceProbability method in each subclass"
        p = 0.0
        sentence = sen.split()
        for i in range(1,len(sentence)):
            p += self.dist.prob(sentence[i],sentence[i-1])
        return p

# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothBigramModel(LanguageModel):
    def __init__(self, corpus):
        #print "Subtask: implement the smoothed bigram language model"
        self.dist =  SmoothBigramDist(corpus)

    def generateSentence(self):
        #print "Implementing generateSentence method for Bigram"
        sent = [start]
        wordi = start
        while(1):
            token = self.dist.draw(wordi)
            if token != end:
                sent.append(token)
                wordi = token
            else:  
                break
        sent.append(end)
        return " ".join(sent)

    # Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        #print "Implementing the getSentenceProbability method in each subclass"
        p = 0.0
        sentence = sen.split()
        for i in range(1,len(sentence)):
            p += self.dist.prob(sentence[i],sentence[i-1])
        return p

    def getCorpusPerplexity(self, corpus):
        logp = []
        N = 0
        for i in range(len(corpus)):
            for j in range(1,len(corpus[i])):
                #print self.dist.prob(corpus[i][j],corpus[i][j-1])
                logp.append( self.dist.prob(corpus[i][j],corpus[i][j-1] ) ) 
                N += 1
        sump = sum(logp) / N
        p = pow(math.e, -sump)
        return p

# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0

    # Returns the probability of word in the distribution
    def prob(self, word):
        return math.log(self.counts[word]/self.total)

    # Generate a single random word according to the distribution
    def draw(self):
        rand = self.total*random.random()
        for word in self.counts:
            rand -= self.counts[word]
            if rand <= 0.0:
                return word
# End sample unigram dist code

class BigramDist:
    def __init__(self,corpus):
        self.bicounts = defaultdict(float)
        self.unicounts = defaultdict(float)
        self.train(corpus)

    def train(self,corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                self.unicounts[corpus[i][j]] += 1.0
                if j != 0:    
                    self.bicounts[corpus[i][j],corpus[i][j-1]] += 1.0
    
    def prob(self,word,prev_word):
        return math.log(self.bicounts[word,prev_word]/self.unicounts[prev_word])

    def draw(self,wordi):
        rand = self.unicounts[wordi]*random.random()
        for (word,prev_word) in self.bicounts:
            if wordi == prev_word:
                rand -= self.bicounts[word,prev_word]
                if rand <= 0.0:
                    return word

class SmoothBigramDist:
    def __init__(self,corpus):
        self.bicounts = defaultdict(float)
        self.unicounts = defaultdict(float)
        self.D = 0.0
        self.total = 0.0
        self.train(corpus)
        self.computeD()
        self.Sw = defaultdict(float)
        self.computeSw()

    def train(self,corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                self.unicounts[corpus[i][j]] += 1.0
                if j != 0:    
                    self.bicounts[corpus[i][j],corpus[i][j-1]] += 1.0
                self.total += 1.0

    def computeD(self):
        once = sum( x == 1 for x in self.bicounts.values() )
        twice  = sum( x == 2 for x in self.bicounts.values() )
        self.D = float(once) / ( once + (2 * twice) )
    
    def computeSw(self):
        for (wordi,wordi_1) in self.bicounts:
            self.Sw[wordi_1] += 1 

    def prob(self,word,prev_word):
        pad = max(self.bicounts[word,prev_word] - self.D, 0) / self.unicounts[prev_word]
        pad += ( ( self.D / self.unicounts[prev_word] ) * self.Sw[prev_word] * (self.unicounts[word] / (self.total - self.unicounts[start]) ) )
        return math.log(pad)

    def draw(self,wordi):
        rand = self.unicounts[wordi]*random.random()
        for key in self.unicounts:
            if key == start:
                continue
            rand -= (max(self.bicounts[key,wordi] - self.D, 0) + (self.D * self.Sw[wordi] * (self.unicounts[key] / (self.total - self.unicounts[start]) ) ) )
            if rand <= 0.0:
                return key

def Vocab(corpus):
    vocab = []
    for sen in corpus:
        vocab.extend(sen)
    vocab = list(set(vocab))
    vocab.remove(start)
    return vocab

def preprocessTest(corpus, vocab):
    for i in range(len(corpus)):
        corpus[i] = [start]+corpus[i]+[end]
        for j in range(1,len(corpus[i])-1):
            if(corpus[i][j] not in vocab):
                corpus[i][j] = UNK
    return corpus
#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    vocab = Vocab(trainCorpus)    
    # Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print "Sample UnigramDist output:"
    print "Probability of \"vader\": ", unigramDist.prob("vader")
    print "Probability of \""+UNK+"\": ", unigramDist.prob(UNK)
    print "\"Random\" draw: ", unigramDist.draw()
    # Sample test run for unigram model
    unigram = UnigramModel(trainCorpus)
    # Generate 20 sentences from Unigram Model
    unigram.generateSentencesToFile(20, "unigram_output.txt")

    # Run for bigram
    bigramDist = BigramDist(trainCorpus)
    print "Sample BigramDist output:"
    print "Probability of \"johnny depp\": ", bigramDist.prob("depp","johnny")

    bigram = BigramModel(trainCorpus)
    # Generate 20 sentences from Bigram Model
    bigram.generateSentencesToFile(20, "bigram_output.txt")
    # Run for Smoothed bigram
    smoothbigramDist = SmoothBigramDist(trainCorpus)
    print "Sample Smoothed BigramDist output:"
    print "Probability of \"johnny depp\": ", smoothbigramDist.prob("depp","johnny")
    smoothbigram = SmoothBigramModel(trainCorpus)
    smoothbigram.generateSentencesToFile(20, "smooth_bigram_output.txt")
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    posTestCorpus = preprocessTest(posTestCorpus,vocab)
    negTestCorpus = preprocessTest(negTestCorpus,vocab)

    trainPerp = unigram.getCorpusPerplexity(trainCorpus)
    posPerp = unigram.getCorpusPerplexity(posTestCorpus)
    negPerp = unigram.getCorpusPerplexity(negTestCorpus)
   
    print "Unigram: Perplexity of positive training corpus:    "+ str(trainPerp) 
    print "Unigram: Perplexity of positive review test corpus: "+ str(posPerp)
    print "Unigram: Perplexity of negative review test corpus: "+ str(negPerp)

    trainPerp = smoothbigram.getCorpusPerplexity(trainCorpus)
    posPerp = smoothbigram.getCorpusPerplexity(posTestCorpus)
    negPerp = smoothbigram.getCorpusPerplexity(negTestCorpus)   

    print "Smooth Bigram: Perplexity of positive training corpus:    "+ str(trainPerp) 
    print "Smooth Bigram: Perplexity of positive review test corpus: "+ str(posPerp)
    print "Smooth Bigram: Perplexity of negative review test corpus: "+ str(negPerp)

