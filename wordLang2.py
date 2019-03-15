import string
from collections import defaultdict
from math import log

def get_word_bigrams(words_list):
    counts = defaultdict(lambda: defaultdict(int))
    for (x, y) in zip( words_list, words_list[1:] ):
        counts[x][y]+=1
    return counts

def get_bigram_freq_of_freq(counts, freq):
    return sum( 1 for c in counts.values() for x in c.values() if x == freq )

THRESHOLD = 10

def get_total_prob(input_str, counts, total_bigram_count, freq_of_freq):
    total_prob = 0;
    words = input_str.split()

    for (k_1, k) in zip(words, words[1:]):
        bigram_count = counts[k_1][k]
        unigram_count = sum(counts[k_1].values())
        if (bigram_count <= THRESHOLD):
            if bigram_count == 0:
                bigram_n_plus_1_count = freq_of_freq[0]
                prob = float(bigram_n_plus_1_count) / float(total_bigram_count)
            else:
                freq_n_plus_1 = freq_of_freq[bigram_count]  # Nc+1
                freq_n = freq_of_freq[bigram_count - 1]  # Nc
                freq_k_plus_one = freq_of_freq[THRESHOLD]  # Nk+1
                freq_1 = freq_of_freq[1 - 1]  # N1
                c_discount = float(bigram_count + 1) * (float(freq_n_plus_1) / float(freq_n))
                threshold_normalization = ((THRESHOLD + 1) * float(freq_k_plus_one)) / float(freq_1)
                denominator = float(1) - threshold_normalization
                # new disocunted bigram count: c*
                bigram_discount = (c_discount - (bigram_count * threshold_normalization)) / (denominator)
                prob = float(bigram_discount) / float(unigram_count)
        else:
            prob = float(bigram_count) / float(unigram_count)
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
    eng_bigram_counts = get_word_bigrams(eng_tokens)
    total_eng_bigram_count = (len(eng_tokens) - 1) ** 2
    eng_freq_of_freq = [get_bigram_freq_of_freq(eng_bigram_counts, x) for x in range(1, 22)]
    # Generate french language model
    lang_fr_file = open("FR.txt")
    fr_tokens = lang_fr_file.read().translate(string.punctuation).translate(string.digits).lower().split()
    fr_bigram_counts = get_word_bigrams(fr_tokens)
    total_fr_bigram_count = (len(fr_tokens) - 1) ** 2
    fr_freq_of_freq = [get_bigram_freq_of_freq(fr_bigram_counts, x) for x in range(1, 22)]
    # Generate italian language model
    lang_gr_file = open("GR.txt")
    gr_tokens = lang_gr_file.read().translate(string.punctuation).translate(string.digits).lower().split()
    gr_bigram_counts = get_word_bigrams(gr_tokens)
    total_gr_bigram_count = (len(gr_tokens) - 1) ** 2
    gr_freq_of_freq = [get_bigram_freq_of_freq(gr_bigram_counts, x) for x in range(1, 22)]

    test_file = open("LangId.test.txt",'r', encoding='UTF-8')
    solution_file = open("wordLangId2.out", 'w')

    line_number = 1
    for line in test_file.readlines():
        eng_res = get_total_prob(line, eng_bigram_counts, total_eng_bigram_count, eng_freq_of_freq)
        fr_res = get_total_prob(line, fr_bigram_counts, total_fr_bigram_count, fr_freq_of_freq)
        gr_res = get_total_prob(line, gr_bigram_counts, total_gr_bigram_count, gr_freq_of_freq)

        prediction = max(eng_res, fr_res, gr_res)
        if prediction == eng_res:
            solution_file.write(str(line_number) + ". EN\n")
        elif prediction == fr_res:
            solution_file.write(str(line_number) + ". FR\n")
        else:
            solution_file.write(str(line_number) + ". GR\n")
        line_number += 1

    lang_eng_file.close()
    lang_fr_file.close()
    lang_gr_file.close()
    test_file.close()
    solution_file.close()

    print("The accuracy of bigram model with Good Turing smoooth is :")
    get_accuracy("LangID.gold.txt", "wordLangId2.out")



