from utils import *
import unittest

class DictionaryUtilsTest(unittest.TestCase):
    def test_invert_dict(self):
        return NotImplementedError

    def test_update_in_place(self):
        d1 = {'a':[1,2,3]}
        d2 = {'b':[3,4,5], 'a':[1,2,4,5]}
        return NotImplementedError

    def test_merge_dict(self):

        return NotImplementedError

    def test_intersect_dict(self):
        return NotImplementedError

class MetricUtilsTest(unittest.TestCase):
    def test_in_interval(self):
        self.assertEqual(True, in_interval(5, 1, 10))
        self.assertEqual(False, in_interval(1, 5, 10))
        self.assertEqual(True, in_interval(1.1, 1, 2))
        self.assertEqual(False, in_interval(0.9, 1, 2))
    
def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DictionaryUtilsTest))
    suite.addTests(unittest.makeSuite(MetricUtilsTest))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite', verbosity=2)

