import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class that stores viterbi and its backpointer
class Viterbi:
    def __init__(self):
        self.viterbi = 0
        self.backpointer = ''

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences                
            return sens
        else:
            print "Error: unlabeled data file ", f, " does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print "Error: unlabeled data file ", f, " does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script  
### End file I/O methods ###  

    # Constructor
    def __init__(self, unknownWordThreshold=5): 
        # Initialize the rest of your data structures here ###
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        self.tag_tagcounts = defaultdict(float)
        self.tag_counts = defaultdict(float)
        self.word_tagcounts = defaultdict(float)
        self.vocab = defaultdict(set)
        self.trellis = []

    # Preprocess for UNK token to allow unseen words
    def preprocess(self, data):
        tokens = defaultdict(float)
        rare_words = defaultdict(list)

        for i in range(len(data)):
            for j in range(len(data[i])):
                if( tokens[data[i][j].word] > (self.minFreq -1) ):
                    tokens[data[i][j].word] += 1.0
                    if data[i][j].word in rare_words:
                        rare_words.pop(data[i][j].word,None)
                else:
                    tokens[data[i][j].word] += 1.0
                    rare_words[data[i][j].word].append((i,j))
                    self.vocab[data[i][j].word].add(data[i][j].tag)

        for word,positions in rare_words.iteritems():
            for (sent,index) in positions:
                data[sent][index].word = UNK
            tokens.pop(word)
            self.vocab.pop(word)
        return data

    # Function to preprocess test data
    def preprocessTest(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if(data[i][j] not in self.vocab):
                    data[i][j] = UNK
        return data 

    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        data = self.preprocess(data)
        for i in range(len(data)):
            self.tag_tagcounts[data[i][0].tag,'start'] += 1.0
            self.tag_counts[data[i][0].tag] += 1.0
            self.tag_counts['start'] += 1.0
            self.word_tagcounts[data[i][0].word,data[i][0].tag] += 1.0
            for j in range(1,len(data[i])):
                self.tag_tagcounts[data[i][j].tag,data[i][j-1].tag] += 1.0
                self.tag_counts[data[i][j].tag] += 1.0
                self.word_tagcounts[data[i][j].word,data[i][j].tag] += 1.0
            self.tag_tagcounts['end',data[i][0].tag] += 1.0
    
    # Function to obtain the transmission probabilities                
    def trans_prob(self, tag, prev_tag):
        return log( (self.tag_tagcounts[tag,prev_tag] + 1) / (self.tag_counts[prev_tag] + (len(self.tag_counts)-1)) )

    # Function to obtain the emission probabilities
    def emiss_prob(self, word, tag):
        if self.word_tagcounts[word,tag] == 0 :
            return float("-inf")
        return log(self.word_tagcounts[word,tag]/self.tag_counts[tag])

    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        #data = self.preprocessTest(data)
        file=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            #print senString
            print >>file, senString.rstrip()
        
    def unpack(self, n, t, tags):
        i = n;
        tags_seq = []
        while (i > 0):
            tags_seq.insert(0, tags[t])
            t = self.trellis[t][i-1].backpointer
            i -= 1
        return tags_seq

    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags 
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        self.trellis = [[Viterbi() for i in range(len(words))] for j in range(len(self.tag_counts)-1)] 
        tags = self.tag_counts.keys()
        tags.remove('start')
        if(words[0] not in self.vocab):
            words[0] = UNK
        # returns the list of Viterbi POS tags (strings)
        for i in range(len(tags)):
            # INITIALIZATION
            self.trellis[i][0].viterbi = self.trans_prob(tags[i],'start') + self.emiss_prob(words[0],tags[i])
        for t in range(1,len(words)):
            if(words[t] not in self.vocab):
                words[t] = UNK
            for j in range(len(tags)):
                self.trellis[j][t].viterbi = float("-inf")
                for k in range(len(tags)):
                    tmp = self.trellis[k][t-1].viterbi + self.trans_prob(tags[j],tags[k])
                    if (tmp > self.trellis[j][t].viterbi):
                        self.trellis[j][t].viterbi = tmp
                        self.trellis[j][t].backpointer = k
                self.trellis[j][t].viterbi += self.emiss_prob(words[t],tags[j])
        t_max = 0
        t = len(words)-1
        vit_max = float("-inf")
        for j in range(len(tags)):
            if (self.trellis[j][t].viterbi + self.trans_prob('end',tags[j]) > vit_max):
                t_max = j
                vit_max = self.trellis[j][t].viterbi + self.trans_prob('end',tags[j])
        return self.unpack(len(words), t_max, tags)

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
