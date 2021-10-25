from preprocessing import load_data, load_unprocessed_data, STOPWORDS, ADJ, ADV, PRE_EMBEDDING_FILTERS, process_string, crop_text
from itertools import chain
import unittest
import string
import numpy as np


class DefaultFilterTest(unittest.TestCase):
    def test_default_filter(self):
        test_df = load_data('data/250_test_data.csv', 'job_title', 'job_description', 0)
        test_vocab = np.unique([*chain.from_iterable([v.split(' ') for v in test_df.values()])]).tolist()
        self.assertEqual([], [e for e in test_vocab if e.islower() == False])
        self.assertEqual([], [e for e in test_vocab if e.isalpha() == False])
        self.assertEqual([], [e for e in test_vocab if e.isnumeric() == True])
        self.assertEqual([], [e for e in test_vocab if string.punctuation in e])
        self.assertNotIn(STOPWORDS, test_vocab)
        self.assertNotIn(ADJ, test_vocab)
        self.assertNotIn(ADV, test_vocab)


class ExtractionFilterTest(unittest.TestCase):
    def test_extraction_filter(self):
        test_df = load_unprocessed_data('data/250_test_data.csv', 'job_title', 'job_description')
        test_vocab = [*chain.from_iterable([v.split(' ') for v in test_df.values()])]
        self.assertNotEqual([], [e for e in test_vocab if 'NUM' in e])
        self.assertNotEqual([], [e for e in test_vocab if 'NOUN' in e])

class OtherFilterTest(unittest.TestCase):
    def test_whitespace_replace(self):
        self.assertIn('_', process_string('software engineer', PRE_EMBEDDING_FILTERS))
        self.assertNotIn(' ', process_string('software engineer', PRE_EMBEDDING_FILTERS))

    def test_crop_text(self):
        def get_word_list(n):
            return ' '.join(['word'] * n)
        self.assertEqual(8 , len(crop_text(get_word_list(10), 0.1).split(' ')))
        self.assertEqual(100, len(crop_text(get_word_list(100), 0).split(' ')))
        self.assertEqual(50, len(crop_text(get_word_list(100), 0.25).split(' ')))


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DefaultFilterTest))
    suite.addTests(unittest.makeSuite(ExtractionFilterTest))
    suite.addTests(unittest.makeSuite(OtherFilterTest))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite', verbosity=2)
