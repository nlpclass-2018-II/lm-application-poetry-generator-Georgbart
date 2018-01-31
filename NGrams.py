import nltk
import re
import math
import operator
from nltk import RegexpTokenizer
from collections import Counter

    #First input & Tokenize
def startToken(input):
    tokenizer = RegexpTokenizer(r'(\w+)')
    file = open(input,'r')
    text = file.read()
    tokens = tokenizer.tokenize(text.lower())
    file.close()
    return tokens

#Further input & tokenize
def getToken(tokens,input):
    tokenizer = RegexpTokenizer(r'(\w+)')
    file = open(input,'r')
    text = file.read()
    tokens += tokenizer.tokenize(text.lower())
    file.close()

#Build Unigram table
def getUnigram(tokens):
    unigram = {}
    unigram_count = 0
    for token in tokens:
        if token in unigram:
            unigram[token] += 1
        else:
            unigram[token] = 1
            unigram_count += 1
    return unigram, unigram_count

#Build Bigram table
def getBigram(tokens):
    length = len(tokens)
    bigram = {}
    bigram_count = 0
    for x in range(0, length - 1):
        if tokens[x] in bigram:
            if tokens[x + 1] in bigram[tokens[x]]:
                bigram[tokens[x]][tokens[x + 1]] += 1
            else:
                bigram[tokens[x]][tokens[x + 1]] = 1
                bigram_count += 1
        else:
            bigram[tokens[x]] = {}
            bigram[tokens[x]][tokens[x + 1]] = 1
            bigram_count += 1
    return bigram, bigram_count

#Handle unseen word
def getGT(tokens, unigram_count):
    bigram_max = unigram_count * unigram_count
    bigram, bigram_count = getBigram(tokens)
    bigram_unseen = bigram_max - bigram_count

    num_of_gram_of_counts = [bigram_unseen, 0, 0, 0, 0, 0]
    for first in bigram:
        for second in bigram[first]:
            counter = bigram[first][second]
            if counter < 6:
                num_of_gram_of_counts[counter] += 1

    GT_counts = [0, 0, 0, 0, 0]
    for x in range(0, 5):
        GT_counts[x] = (x + 1) * (num_of_gram_of_counts[x + 1] * 1.0000) / (num_of_gram_of_counts[x] * 1.0000)
    return GT_counts

#Calculate Probability
def probability(first, second, unigram, bigram, gt_counts):
    counter = 0
    if second.lower() in bigram[first.lower()]:
        counter = bigram[first.lower()][second.lower()]
    if counter < 5:
        counter = gt_counts[counter]
    return counter / (unigram[first.lower()] * 1.0)

# Calculate perplexity
def compute_perplexity(tester, trainer1, trainer2):
    tokens_tester = startToken(tester)
    tokens_trainer = startToken(trainer1)
    getToken(tokens_trainer, trainer2)

    unigram, ucount = getUnigram(tokens_trainer)
    bigram, bcount = getBigram(tokens_trainer)
    gt_count = getGT(tokens_trainer, ucount)

    tokens_tester_count = len(tokens_tester)

    unk = 'unknown'
    length = len(tokens_tester)
    for i in range(0, length, 1):
        if tokens_tester[i] not in unigram:
            tokens_tester[i] = unk
            if unk not in unigram:
                unigram[unk] = 1
            else:
                unigram[unk] += 1
            if 'unknown' not in bigram:
                bigram[unk] = {}
                bigram[unk][unk] = 1

            else:
                bigram[unk][unk] += 1

    total = 0
    for x in range(0, length - 1):
        prob = probability(tokens_tester[x], tokens_tester[x + 1], unigram, bigram, gt_count)
        total += math.log(prob) * (-1)

    return math.exp((total / length))


# Generate Poetry
def generatePoetry(trainer1, trainer2, maxwordcount):
    token = startToken(trainer1)
    getToken(token, trainer2)

    unigram, ucount = getUnigram(tokens)
    bigram, bcount = getBigram(tokens)
    gt_count = getGT(tokens, ucount)

    sentences = ''
    wordcount = 0
    lastWord = ''
    lastProb = 0
    currentProb = 0
    maxProb = 0

    while (wordcount < maxwordcount):
        if (wordcount == 0):
            for word, count in unigram.items():
                currentProb = count / len(token)
                if maxProb < currentProb:
                    maxProb = currentProb
                    lastWord = word

            sentences += lastWord
            lastProb = maxProb
            wordcount += 1

        else:
            if lastWord not in bigram:
                maxProb = 0
                for word, count in unigram.items():
                    currentProb = count / len(token)
                    if maxProb < currentProb:
                        maxProb = currentProb
                        lastWord = word

            else:
                maxProb = 0
                for word in bigram[lastWord]:
                    currentProb = lastProb * probability(lastWord, word, unigram, bigram, gt_count)
                    if maxProb < currentProb:
                        maxProb = currentProb
                        lastWord = word

            sentences += ' '
            sentences += lastWord
            lastProb = maxProb
            wordcount += 1

    return sentences
