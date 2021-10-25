from pattern_learning import discover_patterns, extract_patterns, validate_multi_terms, get_c_value, get_n_words, get_est_recall 
from preprocessing import load_unprocessed_data
from itertools import chain

import unittest


TEST_SEED_LIST = ['NOUNpython', 'NOUNc', 'NOUNstatistics', 'NOUNnetworks', 'NOUNai']
TEST_DF = load_unprocessed_data('data/250_test_data.csv', 'job_title', 'job_description')

class DiscoverPatternsTest(unittest.TestCase):
    def test_found_patterns_pattern_discovery(self):
        test_pattern_list, test_val_dict = discover_patterns(TEST_DF, TEST_SEED_LIST)
        self.assertIn(['PREPwith ADJextensive NOUNknowledge PREPof \\b[a-zA-Z]+', 'last'], test_pattern_list)
        self.assertIn(['NOUNknowledge PREPof NOUNpython CCONand/or \\b[a-zA-Z]+', 'last'], test_pattern_list)
        self.assertIn(['\\b[a-zA-Z]+ CCONand/or NOUNnetworks', 'first'], test_pattern_list)
        self.assertIn(['NOUNknowledge PREPof \\b[a-zA-Z]+', 'last'], test_pattern_list)
        self.assertIn(['NOUNstatistics CCONand/or \\b[a-zA-Z]+', 'last'], test_pattern_list)
        
    def test_not_found_patterns_pattern_discovery(self):
        test_pattern_list, test_val_dict = discover_patterns(TEST_DF, TEST_SEED_LIST)
        self.assertNotIn(['PREPwe VERBare VERBlooking PREPfor \\b[a-zA-Z]+', 'last'], test_pattern_list)
        self.assertNotIn(['\\b[a-zA-Z]+ NOUNknowledge PREPof NOUNstatistics CCONand/or NOUNai', 'first'], test_pattern_list)
        self.assertNotIn(['VERBare VERBlooking PREPfor \\b[a-zA-Z]+', 'last'], test_pattern_list)
        self.assertNotIn(['\\b[a-zA-Z]+ NOUNknowledge PREPof', 'first'], test_pattern_list)
        self.assertNotIn(['NOUNsomeone PREPwith ADJextensive \\b[a-zA-Z]+'], test_pattern_list)

    def test_minimum_and_maximum_length_patterns_pattern_discovery(self):
        test_pattern_list, test_val_dict = discover_patterns(TEST_DF, TEST_SEED_LIST)
        self.assertGreater(min([len(pattern[0].split(' ')) for pattern in test_pattern_list]), 2)
        self.assertLess(max([len(pattern[0].split(' ')) for pattern in test_pattern_list]), 6)
    
    def test_pattern_list_not_empty(self):
        test_pattern_list, test_val_dict = discover_patterns(TEST_DF, TEST_SEED_LIST)
        self.assertNotEqual([], test_pattern_list)

class ExtractPatternsTest(unittest.TestCase):
    def test_pattern_extraction_with_no_input_pattern_list(self):
        test_pattern_list = []
        self.assertNotEqual([], [*chain.from_iterable([v for v in extract_patterns(TEST_DF, test_pattern_list).values()])])

class ValidationTest(unittest.TestCase):
    def test_validate_multi_terms(self):
        return NotImplementedError

    def test_get_c_value(self):
        return NotImplementedError

    def test_validate_patterns(self):
        test_pattern_list, test_val_dict = discover_patterns(TEST_DF, TEST_SEED_LIST)
        test_validated_pattern_list = validate_patterns(test_val_dict, test_pattern_list, TEST_SEED_LIST)
        print(test_validated_pattern_list)

    def test_get_n_words(self):
        self.assertEqual(get_n_words('knowledge of python'), 3)
        self.assertEqual(get_n_words('must know how to deal with complex problems and/or mathematical equations'), 11)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DiscoverPatternsTest))
    suite.addTests(unittest.makeSuite(ExtractPatternsTest))
    suite.addTests(unittest.makeSuite(ValidationTest))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite', verbosity=2)
