import os.path
import sys
from operator import itemgetter
from collections import defaultdict

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# A class for evaluating POS-tagged data
class Eval:
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

    def __init__(self, goldFile, testFile):
        self.gold_data = self.readLabeledData(goldFile)
        self.test_data = self.readLabeledData(testFile)

    def getTokenAccuracy(self):
        count = 0.0
        total = 0.0
        for i in range(len(self.test_data)):
            total += len(self.test_data[i])
            for j in range(len(self.test_data[i])):
                if( self.test_data[i][j].tag == self.gold_data[i][j].tag ):
                    count += 1.0
        return count/total
    
    def getSentenceAccuracy(self):
        count = 0.0
        total = len(self.test_data)
        for i in range(len(self.test_data)):
            incount = 0.0 
            for j in range(len(self.test_data[i])):
                if( self.test_data[i][j].tag == self.gold_data[i][j].tag ):
                    incount += 1.0
            if (incount == len(self.test_data[i])):
                count += 1.0
        return count/total
    
    # Write a confusion matrix to file
    def writeConfusionMatrix(self, outFile):
        tag_pos = defaultdict(list)
        for i in range(len(self.gold_data)):
            for j in range(len(self.gold_data[i])):
                tag_pos[self.gold_data[i][j].tag].append((i,j))
        keys = tag_pos.keys()
        counts = defaultdict(int)
        for i in range(len(keys)):
            for j in range(len(keys)):
                counts[keys[i],keys[j]] = 0
        for tag,positions in tag_pos.iteritems():
            for i,j in positions:
                if self.gold_data[i][j].tag != self.test_data[i][j].tag :
                    counts[self.gold_data[i][j].tag,self.test_data[i][j].tag] += 1
        outfile = open(outFile, "w")
        outfile.write('%4s' %"-" + " ")
        outfile.write(" ".join('%4s' %x for x in keys[0:len(keys)-18]))
        outfile.write("\n")
        for i in range(len(keys)):
            outfile.write('%4s' %keys[i] +" ")
            for j in range(len(keys)-18):
                outfile.write( '%4s' %str(counts[keys[i],keys[j]]) + " ")
            outfile.write("\n")
        outfile.write(140*'#') #140 characters for vim
        outfile.write("\n")
        outfile.write(140*'#') #140 characters for vim
        outfile.write('%4s' %"-" + " ")
        outfile.write(" ".join('%4s' %x for x in keys[len(keys)-18:]))
        outfile.write("\n")
        for i in range(len(keys)):
            outfile.write('%4s' %keys[i] +" ")
            for j in range(len(keys)-18,len(keys)):
                outfile.write( '%4s' %str(counts[keys[i],keys[j]]) + " ")
            outfile.write("\n")
        outfile.close()

    # Return the tagger's precision on predicted tag t_i
    def getPrecision(self, tagTi):

        gold_count = 0.0
        test_count = 0.0
        for i in range(len(self.gold_data)):
            for j in range(len(self.gold_data[i])):
                if self.test_data[i][j].tag == tagTi:
                    test_count += 1.0
                    if self.gold_data[i][j].tag == tagTi:
                        gold_count += 1.0
        return gold_count/test_count

    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        gold_count = 0.0
        test_count = 0.0
        for i in range(len(self.gold_data)):
            for j in range(len(self.gold_data[i])):
                if self.gold_data[i][j].tag == tagTj:
                    gold_count += 1.0
                    if self.test_data[i][j].tag == tagTj:
                        test_count += 1.0
                    
        return test_count/gold_count
    

if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    gold = sys.argv[1]
    test = sys.argv[2]
    # You need to implement the evaluation class
    eval = Eval(gold, test)
    # Calculate accuracy (sentence and token level)
    print "Token accuracy: ", eval.getTokenAccuracy()
    print "Sentence accuracy: ", eval.getSentenceAccuracy()
    # Calculate recall and precision
    print "Precision on tag NNP: ", eval.getPrecision('NNP')
    print "Recall for tag NNP: ", eval.getRecall('NNP')
    # Write a confusion matrix
    eval.writeConfusionMatrix("conf_matrix.txt")
