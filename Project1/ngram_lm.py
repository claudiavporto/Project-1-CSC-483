import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c


def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    ngram_text = start_pad(c) + text
    ngram_list = []
    for i in range(len(ngram_text) - c):
        context = ngram_text[i:i + c]
        char = ngram_text[i + c]
        ngram = (context, char)
        ngram_list.append(ngram)

    return ngram_list


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.__c = c
        self.__k = k
        self.__vocab = set()
        self.__ngrams = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.__vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            self.__vocab.update(char)

        ngram_list = ngrams(self.__c, text)
        for ngram in ngram_list:
            if ngram not in self.__ngrams:
                self.__ngrams[ngram] = 1
            else:
                self.__ngrams[ngram] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        ngram = context, char
        ngram_count = 0
        context_count = self.__context_counter(context)
        if context_count != 0:
            if ngram in self.__ngrams:
                ngram_count = self.__ngrams[ngram]
            return (ngram_count + self.__k) / (context_count + (self.__k * len(self.__vocab)))
        return 1 / len(self.__vocab)

    def __context_counter(self, context):
        '''Returns the count of all ngrams that begin with context char'''
        context_counter = 0
        for ngram, count in self.__ngrams.items():
            if ngram[0] == context:
                context_counter += count

        return context_counter

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        vocab_sort = sorted(self.__vocab)
        r = random.random()
        prob = 0
        i = 0
        while prob <= r and i < len(vocab_sort):
            prob += self.prob(context, vocab_sort[i])
            i += 1

        return vocab_sort[i - 1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = start_pad(self.__c)
        for i in range(length):
            context = text[len(text) - self.__c:]
            text += self.random_char(context)
        return text[self.__c:]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        ngram_list = ngrams(self.__c, text)
        n = len(text)
        corpus_prob = 0
        for ngram in ngram_list:
            prob = self.prob(ngram[0], ngram[1])
            if prob != 0:
                corpus_prob += math.log(prob)
            else:
                return float('inf')
        perplexity = math.exp(-(1/n) * corpus_prob)

        return perplexity

    def set_k(self, k):
        '''sets the value of k for add-k smoothing'''
        self.__k = k

    def prob_str(self, text):
        '''returns the n-gram probability of a string'''
        ngram_list = ngrams(self.__c, text)
        n = len(text)
        str_prob = 0
        for ngram in ngram_list:
            prob = self.prob(ngram[0], ngram[1])

            if prob != 0:
                str_prob += math.log(prob)
            else:
                return float('inf')

        return str_prob

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c, k)
        self.__lambdas = []
        for i in range(c + 1):
            self.__lambdas.append(1/(c + 1))

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.__vocab

    def update_lambdas(self, lambdas: list):
        'set the lambda values for interpolation'
        self.__lambdas = lambdas

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            if char not in super().get_vocab():
                self._NgramModel__vocab.update(char)

        order_ngrams = []
        for i in range(self._NgramModel__c + 1):
            ngram_list = ngrams(i, text)
            order_ngrams.append(ngram_list)

        for ngram_list in order_ngrams:
            for ngram in ngram_list:
                if ngram not in self._NgramModel__ngrams:
                    self._NgramModel__ngrams[ngram] = 1
                else:
                    self._NgramModel__ngrams[ngram] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        prob = 0
        for i in range(self._NgramModel__c + 1):
            prob_holder = super().prob(context[len(context) - i:], char) * self.__lambdas[i]
            prob += prob_holder

        return prob



    ################################################################################
# Your N-Gram Model Experimentations
################################################################################


def NgramModel_tests():
    '''NgramModel class tests'''
    m = NgramModel(1,0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print(m.prob('a', 'b'), '(expected: 1.0)')
    print(m.prob('~', 'c'), '(expected: 0.0)')
    print(m.prob('b', 'c'), '(expected: 0.5)')
    random.seed(1)
    print(m.random_text(25))

    j = NgramModel(0, 0)
    j.update('abab')
    j.update('abcd')
    random.seed(1)
    print([j.random_char('') for i in range(25)])

    n = NgramModel(1,1)
    n.update('abab')
    n.update('abcd')
    print(n.prob('a', 'a'), '(expected: 0.14285714285714285)')
    print(n.prob('a', 'b'), '(expected: 0.5714285714285714)')
    print(n.prob('c', 'd'), '(expected: 0.4)')
    print(n.prob('d', 'a'), '(expected: 0.25)')

    d = NgramModel(1, 0)
    d.update('abab')
    d.update('abcd')
    print(d.perplexity('abcd'), '(expected: 1.189207115002721)')
    print(d.perplexity('abca'), '(expected: inf)')
    print(d.perplexity('abcda'), '(expected: 1.515716566510398)')


def NgramModel_Shakespeare_tests():
    ''' NgramModel class tests on Shakespeare text files'''

    n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print(n.random_text(250))

    n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print(n.random_text(250))

    n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print(n.random_text(250))

    n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print(n.random_text(250))

    shakespeare = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    shakespeare.set_k(1)
    print('---k=1---c=2')
    print("shakes perp: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare.perplexity('nytimes_article.txt'))
    shakespeare.set_k(2)
    print('---k=2---c=2')
    print("shakes perp: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare.perplexity('nytimes_article.txt'))
    shakespeare.set_k(3)
    print('---k=3---c=2')
    print("shakes perp: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare.perplexity('nytimes_article.txt'))
    shakespeare.set_k(4)
    print('---k=4---c=2')
    print("shakes perp: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare.perplexity('nytimes_article.txt'))

    shakespeare2 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    shakespeare2.set_k(1)
    print('---k=1---c=3')
    print("shakes perp: ", shakespeare2.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare2.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare2.perplexity('nytimes_article.txt'))
    shakespeare2.set_k(2)
    print('---k=2---c=3')
    print("shakes perp: ", shakespeare2.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare2.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare2.perplexity('nytimes_article.txt'))
    shakespeare2.set_k(3)
    print('---k=3---c=3')
    print("shakes perp: ", shakespeare2.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare2.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare2.perplexity('nytimes_article.txt'))
    shakespeare2.set_k(4)
    print('---k=4---c=3')
    print("shakes perp: ", shakespeare2.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare2.perplexity('shakespeare_sonnets.txt'))
    print("nyt perp: ", shakespeare2.perplexity('nytimes_article.txt'))


def NgramModelInterpolation_tests():
    '''NgramModelInterpolation class tests'''
    g = NgramModelWithInterpolation(1, 0)
    g.update('abab')
    print(g.prob('a', 'a'), '(expected: 0.25)')
    print(g.prob('a', 'b'), '(expected: 0.75)')
    g.update_lambdas([0.25, 0.25])
    print(g.prob('a', 'a'))
    print(g.prob('a', 'b'))
    g.update_lambdas([0.2, 0.3])
    print(g.prob('a', 'a'))
    print(g.prob('a', 'b'))

    p = NgramModelWithInterpolation(2, 1)
    p.update('abab')
    p.update('abcd')
    print(p.prob('~a', 'b'), '(expected: 0.4682539682539682)')
    print(p.prob('ba', 'b'), '(expected: 0.4349206349206349)')
    print(p.prob('~c', 'd'), '(expected: 0.27222222222222225)')
    print(p.prob('bc', 'd'), '(expected: 0.3222222222222222)')


def LanguageIdentification_tests():
    '''Language identification models tests'''
    af = create_ngram_model_lines(NgramModel, 'train/af.txt')
    cn = create_ngram_model_lines(NgramModel, 'train/cn.txt')
    de = create_ngram_model_lines(NgramModel, 'train/de.txt')
    fi = create_ngram_model_lines(NgramModel, 'train/fi.txt')
    fr = create_ngram_model_lines(NgramModel, 'train/fr.txt')
    ind = create_ngram_model_lines(NgramModel, 'train/in.txt')
    ir = create_ngram_model_lines(NgramModel, 'train/ir.txt')
    pk = create_ngram_model_lines(NgramModel, 'train/pk.txt')
    za = create_ngram_model_lines(NgramModel, 'train/za.txt')

    country_codes = {af: 'af', cn: 'cn', de: 'de', fi: 'fi', fr: 'fr', ind: 'in',
                         ir: 'ir', pk: 'pk', za: 'za'}

    val_texts = {'val/af.txt': 'af', 'val/cn.txt': 'cn', 'val/de.txt': 'de', 'val/fi.txt': 'fi',
                 'val/fr.txt': 'fr', 'val/in.txt': 'in', 'val/ir.txt': 'ir', 'val/pk.txt': 'pk',
                 'val/za.txt': 'za'}

    val_accuracies = {'val/af.txt': 0, 'val/cn.txt': 0, 'val/de.txt': 0, 'val/fi.txt': 0,
                      'val/fr.txt': 0, 'val/in.txt': 0, 'val/ir.txt': 0, 'val/pk.txt': 0,
                      'val/za.txt': 0}

    for val_txt in val_texts.keys():
        predicted_correct = 0
        actual_total = 0
        correct_country = val_texts[val_txt]
        file = open(val_txt, 'r')
        cities = file.read().splitlines()
        for city in cities:
            actual_total += 1
            highest_prob = 0
            highest_prob_country = ''
            for model in country_codes.keys():
                current_prob = model.prob_str(city)
                if current_prob != float('inf'):
                    if (abs(current_prob) < abs(highest_prob)) or highest_prob == 0:
                        highest_prob = current_prob
                        highest_prob_country = country_codes[model]
                elif highest_prob == 0:
                    highest_prob_country = country_codes[model]
            if highest_prob_country == correct_country:
                predicted_correct += 1
        accuracy = predicted_correct / actual_total
        val_accuracies[val_txt] = accuracy

    print(val_accuracies)


if __name__ == "__main__":

    NgramModel_tests()
    NgramModel_Shakespeare_tests()
    NgramModelInterpolation_tests()
    LanguageIdentification_tests()



