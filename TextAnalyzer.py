# Analyze Text with a TextAnalyzer object!
#
# By: Renuka Murthi and Annie Rauwerda

import unittest  # import the library needed for testing
import math
import csv
import os


class TextAnalyzer:

    def __init__(self, filepath):
        """Initializes the TextAnalyzer object, using the file at filepath.
        Initialize the following instance variables: filepath (string),
        lines (list)"""
        self.filepath = filepath
        lines = []

    def sentence_count(self):
        """Returns the number of sentences in the file (seperated by .)
        Note that if there are no '.' in the sentences return 1"""
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        count = 0
        for x in lynes:
            for y in x:
                lst = y.split()
                for x in lst:
                    if x == '.':
                        count += 1
        if count == 0:
            count += 1
        return count


    def words(self):
        """Returns a list of words without punctuation and all lower case.
        For example : 'Cat!' should be 'cat'."""
        # Uncomment the next line

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        return lst




    def remove_stopwords(self, words):
        """This takes in the list of words that are not punctuated and are lowercase.
        Returns a list of words with the stopwords provided by the file 
        stopwords.txt removed. """

        fyle = open('stopwords.txt', 'r')
        lynes = fyle.readlines()
        lynes2 = []
        for x in lynes:
            x = x.replace("\n", '')
            lynes2.append(x)
        lst = []
        for x in words:
            if x not in lynes2:
                lst.append(x)
        return lst


    def word_count(self):
        """Returns the number of words in the file not including the stopwords. A word is defined as any
        text that is separated by whitespace (spaces, newlines, or tabs)."""
        
        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        count = 0
        for x in lst:
            if x not in laines2:
                count += 1
        return count


        


    def vocabulary(self):
        """Returns a list of the unique words in the text, sorted in
        alphabetical order. Capitalization, punctuation, and stopwords should be ignored, so 'Cat!' is the
        same word as 'cat'. The returned words should be all lowercase, without punctuation or stopwords."""

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2 and x not in lst2:
                lst2.append(x)
        lst3 = sorted(lst2, key=lambda x: x[0])
        return lst3



    def frequencies(self):
        """Returns a dictionary of the words in the text and the count of how
        many times they appear. The words are the keys, and the counts are the
        values. All the words should be lower case, without punctuation and does not include stopwords. The order of the keys
        doesn't matter."""

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2:
                lst2.append(x)

        dic = {}
        for x in lst2:
            if x in dic:
                dic[x] += 1
            else:
                dic[x] = 1

        return dic

    def frequency_of(self, word):
        """Returns the number of times the word appears in the text. Capitalization, punctuation, and stopwords
        should be ignored, so 'Cat!' is the same word as 'cat'. If the word does not exist in the text,
        then return 0"""

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2:
                lst2.append(x)

        count = 0
        for x in lst2:
            if x == word:
                count += 1

        return count

    def percent_frequencies(self):
        """Returns a dictionary of the words in the text and the frequency of the
        words as a percentage of the text. The words are the keys, and the
        counts are the values. All the words should be lowercase, without punctuation or stopwords. The order
        of the keys doesn't matter."""

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2:
                lst2.append(x)

        total = len(lst2)
        dic = {}
        for x in lst2:
            if x in dic:
                dic[x] += 1
            else:
                dic[x] = 1

        dic2 = {}
        for x in dic:
            perc = dic[x] / total
            dic2[x] = perc
        
        return dic2


    def most_common(self):
        """Returns the most common word in the text and its frequency in a list.
            There might be a case where multiple words have the same frequency,
            in that case return one of the most common words which should be lowercase,
            without punctuation or stopwords"""
        # Example ouput : ['officer', 6]

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2:
                lst2.append(x)

        dic = {}
        for x in lst2:
            if x in dic:
                dic[x] += 1
            else:
                dic[x] = 1

        maxword = max(dic, key=dic.get)
        lst3 = []
        lst3.append(maxword)
        lst3.append(dic[maxword])

        return lst3

    
    def five_least_common(self):
        """Returns the five least common words in the text and its frequency as a list of tuples.
            If there are not five words in the text, return all the least common words.
            There might be a case where multiple words have the same frequency,
            in that case, return any of the least common words which should be lowercase,
            without punctuation or stopwords"""
        # Example ouput : [(ants', 1), ('apple', 1), ('bat', 1), ('cat', 3)]

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fyle = open(self.filepath, 'r')
        lynes = fyle.readlines()
        lst = []
        for x in lynes:
            words = x.split()
            for y in words:
                y = y.lower()
                for letter in y:
                    if letter in punctuation:
                        y = y.replace(letter, '')
                lst.append(y)
        faile = open('stopwords.txt', 'r')
        laines = faile.readlines()
        laines2 = []
        for x in laines:
            x = x.replace('\n', '')
            laines2.append(x)
        lst2 = []
        for x in lst:
            if x not in laines2:
                lst2.append(x)

        dic = {}
        for x in lst2:
            if x in dic:
                dic[x] += 1
            else:
                dic[x] = 1

        lst3 = []
        for x in range(5):
            minword = min(dic, key=dic.get)
            tupl = (minword, dic[minword])
            lst3.append(tupl)
            dic[minword] = 999999999999999999

        return lst3

        


    def read_sample_csv(self):
        """Reads the sample.csv file and returns the list of fieldnames"""
        # Output Format: filepath, total words, word count removing stopwords, line count, most common word  

        fyle = open('sample.csv', 'r')
        lynes = fyle.readlines()
        lst = lynes[0].split(',')
        lst2 = []
        for x in lst:
            if x[0] == ' ' and x[-1] == '\n':
                lst2.append(x[1:-1])
            elif x[0] == ' ':
                lst2.append(x[1:])
            else:
                lst2.append(x)
        return lst2

    
    def write_analysis_details(self, csvfile):
        """Writes the details of the textual analysis to the csvfile.
        Refer to sample.csv for an example of how this should look.
        Note that for most common word, just write the word and not its frequency"""
        # Output Format: filepath, total words, word count removing stopwords, line count, most common word
        write_file = open(csvfile, 'w')
        filename = self.read_sample_csv()
        writer = csv.writer(write_file)
        writer.writerow(filename)
        writer.writerow([self.filepath, len(self.words()), self.word_count(), self.sentence_count(), self.most_common()[0]])
        write_file.close()




        # fout = open(csvfile, 'w')
        # outputContent = []

        # pathlist = (self.full_path.split("/"))[-2:]
        # filePath = pathLst[0] + '/' + pathLst[1]
        # outputContent.append(filePath)
        # outputContent.append(len(self.words()))
        # woStopwords = self.remove_stopwords(self.words())
        # outputContent.append(len(woStopwords))
        # outputContent.append(len(self.lines))
        # outputContent.append(self.most_common()[0])


        # feelds = ['filepath', 'total_words', 'word_count_without_stop', 'line_count', 'most_common_word']
       

        # with open(csvfile, 'w') as f:
        #     csvw = csv.writer(f, delimiter = ',')
            
        #     csvw.writerow(self.read_sample_csv)
        #     csvw.writerow(row)

        # writer = csv.writer(fyle, delimiter = ',', quotechar='"',  quoting=csv.QUOTE_MINIMAL)
        # fyle.writerow([self.filepath, len(self.words()), self.word_count(), self.lines(), self.most_common])


    
    # Extra Credit!
    # See instructions page
    def similarity_with(self, other_text_analyzer):
        """Extra credit. Calculates the similarity between this text and
        the other text using cosine similarity. Words should be lowercase, withought
        punctiations or stopwords"""
        mag1 = self.most_common()[1]
        mag2 = other_text_analyzer.most_common()[1]

        vocab1 = self.vocabulary()
        vocab2 = other_text_analyzer.vocabulary()

        common_words = []
        for word in vocab1:
            if word in vocab2:
                common_words.append(word)

        freq1= self.frequencies()
        freq2 = other_text_analyzer.frequencies()

        dot = 0
        for word in common_words:
            dot += (freq1[word] * freq2[word])

        final = dot / (mag1 * mag2)
        return final

# These are the tests.

class TestSentenceCount(unittest.TestCase):

    def test_sentence_count_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.sentence_count(), 1)
        self.assertEqual(ta.sentence_count(), 1) # Check that it works when called a second time

    def test_line_count_tiny3(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta.sentence_count(), 3)
        self.assertEqual(ta.sentence_count(), 3) # Check that it works when called a second time

    def test_line_count_the_buckeye_battle_cry(self):
        ta = TextAnalyzer("files_for_testing/buckeye_battle_cry.txt")
        self.assertEqual(ta.sentence_count(), 3)
        self.assertEqual(ta.sentence_count(), 3) # Check that it works when called a second time



class TestWords(unittest.TestCase):

    def test_words_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.words(), ['coffee','is','so', 'good'])

    def test_words_tiny2(self):
        ta2 = TextAnalyzer("files_for_testing/tinyfile_2.txt")
        self.assertEqual(ta2.words(), ['you', 'hate', 'tea'])

    def test_words_tiny3(self):
        ta3 = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        self.assertEqual(ta3.words(), ['i', 'love', 'coffee', 'so', 'so','so','so','so','so','much'])



class TestRemoveStopwords(unittest.TestCase):
    def test_remove_stopwords_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.remove_stopwords(ta.words()), ['coffee', 'good'])
   
    def test_remove_stopwords_tiny3(self):
        ta3 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta3.remove_stopwords(ta3.words()), ['i', 'love', 'coffee', 'much', 'i', 'love', 'tea', 'much', 'i', 'hate', 'juice', 'much'])

    
    def test_remove_stopwords_tiny5(self):
        ta5 = TextAnalyzer("files_for_testing/tinyfile_5.txt")
        self.assertEqual(ta5.remove_stopwords(ta5.words()), ['coffee', 'coffee','coffee', 'ba', 'ba', 'huuuh', 'huuuh', 'blah', 'blah', 'bu', 'bu','bu','bu','bu','howdyy', 'good'])



class TestWordCount(unittest.TestCase):

    def test_word_count_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.word_count(), 2)
        self.assertEqual(ta.word_count(), 2) # Check that it works when called a second time

    def test_word_count_tiny3(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta.word_count(), 12)
        self.assertEqual(ta.word_count(), 12) # Check that it works when called a second time

    def test_word_count_the_osusong(self):
        ta = TextAnalyzer("files_for_testing/osusong.txt")
        self.assertEqual(ta.word_count(), 5)
        self.assertEqual(ta.word_count(), 5) # Check that it works when called a second time



class TestFrequencies(unittest.TestCase):

    def test_frequencies_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.frequencies()['coffee'], 1)
        self.assertEqual(ta.frequencies()['good'], 1)

    def test_frequencies_tiny2(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_2.txt")
        self.assertEqual(ta.frequencies()['you'], 1)
        self.assertEqual(ta.frequencies()['hate'], 1)
        self.assertEqual(ta.frequencies()['tea'], 1)

    def test_frequencies_tiny4(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        self.assertEqual(ta.frequencies()['i'], 1)
        self.assertEqual(ta.frequencies()['love'], 1)
        self.assertEqual(ta.frequencies()['coffee'], 1)
        self.assertEqual(ta.frequencies()['much'], 1)



class TestFrequencyOf(unittest.TestCase):

    def test_frequency_of_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.frequency_of('coffee'), 1)
        self.assertEqual(ta.frequency_of('is'), 0)
        self.assertEqual(ta.frequency_of('so'), 0)
        self.assertEqual(ta.frequency_of('good'), 1)

    def test_frequency_of_osusong(self):
        ta = TextAnalyzer("files_for_testing/osusong.txt")
        self.assertEqual(ta.frequency_of('come'), 1)
        self.assertEqual(ta.frequency_of('on'), 0)
        self.assertEqual(ta.frequency_of('ohio'), 1)
        self.assertEqual(ta.frequency_of('victory'), 1)
        self.assertEqual(ta.frequency_of('through'), 1)

    def test_frequency_of_tiny2(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_2.txt")
        self.assertEqual(ta.frequency_of('you'), 1)
        self.assertEqual(ta.frequency_of('hate'), 1)
        self.assertEqual(ta.frequency_of('tea'), 1)
        self.assertEqual(ta.frequency_of('coffee'), 0)



class TestVocabulary(unittest.TestCase):

    def test_vocabulary_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.vocabulary(), ['coffee', 'good'])

    def test_vocabulary_tiny3(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta.vocabulary(), ['coffee', 'hate', 'i', 'juice', 'love', 'much', 'tea'])

    def test_vocabulary_tiny4(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        self.assertEqual(ta.vocabulary(), ['coffee', 'i', 'love', 'much'])



class TestPercentFrequencyOf(unittest.TestCase):

    def test_percent_frequency_of_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertIn('coffee', ta.percent_frequencies())
        self.assertIn('good', ta.percent_frequencies())
        self.assertAlmostEqual(ta.percent_frequencies()['good'], 1/2)
        self.assertAlmostEqual(ta.percent_frequencies()['coffee'], 1/2)

    def test_percent_frequency_of_tiny3(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertIn('i', ta.percent_frequencies())
        self.assertIn('love', ta.percent_frequencies())
        self.assertIn('coffee', ta.percent_frequencies())
        self.assertIn('much', ta.percent_frequencies())
        self.assertIn('hate', ta.percent_frequencies())
        self.assertIn('juice', ta.percent_frequencies())
        self.assertAlmostEqual(ta.percent_frequencies()['i'], 3/12)
        self.assertAlmostEqual(ta.percent_frequencies()['love'], 2/12)
        self.assertAlmostEqual(ta.percent_frequencies()['coffee'], 1/12)
        self.assertAlmostEqual(ta.percent_frequencies()['tea'], 1/12)
        self.assertAlmostEqual(ta.percent_frequencies()['juice'], 1/12)
        self.assertAlmostEqual(ta.percent_frequencies()['much'], 3/12)
        self.assertAlmostEqual(ta.percent_frequencies()['hate'], 1/12)

    def test_percent_frequency_of_tiny4(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        self.assertAlmostEqual(ta.percent_frequencies()['i'], 1/4)
        self.assertAlmostEqual(ta.percent_frequencies()['love'], 1/4)
        self.assertAlmostEqual(ta.percent_frequencies()['coffee'], 1/4)
        self.assertAlmostEqual(ta.percent_frequencies()['much'], 1/4)



class TestMostCommon1(unittest.TestCase):

    def test_most_common_1_tiny3(self):
        ta3 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta3.most_common()[0], 'i')

    def test_most_common_1_tiny5(self):
        ta5 = TextAnalyzer("files_for_testing/tinyfile_5.txt")
        self.assertEqual(ta5.most_common(), ['bu', 5])

class TestMostCommonMultipleClearCases(unittest.TestCase):

    def test_most_common_multiple_tiny1(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        self.assertEqual(ta.most_common()[1], 1)



class TestFiveLeastCommon(unittest.TestCase):

    def test_five_least_common_tiny3(self):
        ta3 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertEqual(ta3.five_least_common(), [('coffee', 1), ('tea', 1), ('hate', 1), ('juice', 1), ('love', 2)])

    def test_five_least_common_tiny5(self):
        ta5 = TextAnalyzer("files_for_testing/tinyfile_5.txt")
        self.assertEqual(ta5.five_least_common(), [('howdyy', 1), ('good', 1), ('ba', 2), ('huuuh', 2), ('blah', 2)])
        self.assertIsInstance(ta5.five_least_common()[0], tuple)



class TestReadSampleCSV(unittest.TestCase):

    def test_reading_sample_csv(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        self.assertEqual(ta.read_sample_csv(), ['filepath', 'total words', 'word count removing stopwords', 'line count', 'most common word'])



class TestWriteAnalysis(unittest.TestCase):

    def test_write_analysis_details(self):
        ta = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        ta.write_analysis_details('test.csv')
        f = open('test.csv')
        csv_reader = csv.reader(f, delimiter=',')
        lines = [r for r in csv_reader]
        self.assertEqual(ta.read_sample_csv(), ['filepath', 'total words', 'word count removing stopwords', 'line count', 'most common word'])
        self.assertEqual(lines[1], ['files_for_testing/tinyfile_4.txt', '10', '4','1','i'])
        f.close()



class TestSimilarity(unittest.TestCase):
    def test_similarity_1(self):
        ta1 = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        ta2 = TextAnalyzer("files_for_testing/tinyfile_2.txt")
        self.assertAlmostEqual(ta1.similarity_with(ta2), 0.0)

    def test_similarity_2(self):
        ta1 = TextAnalyzer("files_for_testing/tinyfile_1.txt")
        ta2 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertAlmostEqual(ta1.similarity_with(ta2), 0.33333333333333)

    def test_similarity_3(self):
        ta1 = TextAnalyzer("files_for_testing/everycoffee.txt")
        ta2 = TextAnalyzer("files_for_testing/tinyfile_5.txt")
        self.assertAlmostEqual(ta1.similarity_with(ta2), 0.4)


    def test_similarity_4(self):
        ta1 = TextAnalyzer("files_for_testing/tinyfile_4.txt")
        ta2 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
        self.assertAlmostEqual(ta1.similarity_with(ta2), 3)


if __name__ == "__main__":
    # Un-comment this line when you are ready to run the unit tests.
    unittest.main(verbosity=2)

    # You can uncomment out some of these lines to do some simple tests with print statements.
    # Or, use your own print statements here as well!
    # fightsong = TextAnalyzer("files_for_testing/fightsong.txt")
    # print(type(fightsong.five_least_common()[0]))
    # osusong = TextAnalyzer("files_for_testing/osusong.txt")
    # print("Sentence count is ", fightsong.sentence_count())
    # print("Words list is ", fightsong.words())
    # print("Word count is ", fightsong.word_count())
    # print("Vocabulary is ", fightsong.vocabulary())
    # print("Frequencies are ", fightsong.frequencies())
    # print("Most common word and its frequence is ", fightsong.most_common())
    # print("Percent frequencies are ", fightsong.percent_frequencies())

    # ta1 = TextAnalyzer("files_for_testing/tinyfile_1.txt")
    # ta2 = TextAnalyzer("files_for_testing/tinyfile_3.txt")
    # print(ta1.similarity_with(ta2))