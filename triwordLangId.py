import string
from collections import defaultdict
from math import log

def get_word_trigrams(words_list):
    counts = defaultdict(lambda: defaultdict(int))
    for (x, y) in zip((words_list+words_list[1:]), words_list[2:]):
        counts[x][y] += 1
    return counts

def get_total_prob(input_str, counts):
    total_prob = 0;
    words = input_str.split()
    for (k_1, k) in zip(words, words[1:]):
        trigram_count = counts[k_1][k]
        bigram_count = sum(counts[k_1].values())
        # if not bigram_count: print "smoothing ("+ k_1 + ", " + k +")"
        # if not unigram_count: print "smoothing ("+ k_1 +")"
        prob = float(trigram_count + 1) / float(bigram_count + len(counts))
        total_prob += log(prob, 2)  # use log to get the prob
    return total_prob

def get_accuracy(filename1,filename2):
    accuracy = 0
    f1 = open(filename1,'r')
    f2 = open(filename2,'r')
    test = f1.readlines()
    predict = f2.readlines()
    for line in range(len(predict)):
        if test[line+1][-3:-1] == predict[line][-3:-1]:
            accuracy +=1
    print(accuracy/float(len(predict))*100)


if __name__ == '__main__':
    # Generate english language model
    lang_eng_file = open("EN.txt")
    eng_tokens = lang_eng_file.read().translate(string.punctuation).translate(string.digits).lower().split()
    eng_trigram_counts = get_word_trigrams(eng_tokens)
    # Generate french language model
    lang_fr_file = open("FR.txt")
    fr_tokens = lang_fr_file.read().translate(string.punctuation).translate(string.digits).lower().split()
    fr_trigram_counts = get_word_trigrams(fr_tokens)
    # Generate italian language model
    lang_gr_file = open("GR.txt")
    gr_tokens = lang_gr_file.read().translate(string.punctuation).translate(string.digits).lower().split()
    gr_trigram_counts = get_word_trigrams(gr_tokens)

    test_file = open("LangId.test.txt",'r', encoding='UTF-8')
    predict_file = open("triwordLangId.out", 'w')  # output will be printed here

    line_number = 1
    for line in test_file.readlines():
        eng_res = get_total_prob(line, eng_trigram_counts)
        fr_res = get_total_prob(line, fr_trigram_counts)
        gr_res = get_total_prob(line, gr_trigram_counts)

        prediction = max(eng_res, fr_res, gr_res)
        if prediction == eng_res:
            predict_file.write(str(line_number) + ". EN\n")
        elif prediction == fr_res:
            predict_file.write(str(line_number) + ". FR\n")
        else:
            predict_file.write(str(line_number) + ". GR\n")
        line_number += 1

    lang_eng_file.close()
    lang_fr_file.close()
    lang_gr_file.close()
    test_file.close()
    predict_file.close()

    print("The accuracy of trigram model is :")
    get_accuracy("LangID.gold.txt", "triwordLangId.out")