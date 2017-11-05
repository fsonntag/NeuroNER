import unittest

import conlleval


class TestEvaluators(unittest.TestCase):

    input_file = map(lambda l: l.strip(),
     """
    . O O
    . O O
    . O O
    . O O
    a B-PER B-PER
    a I-PER I-PER
    a I-PER I-PER
    a I-PER I-PER
    . O O
    . O O
    . O O
    . O O
    b O B-PER
    b O I-PER
    b O I-PER
    b O I-PER
    . O O
    . O O
    . O O
    . O O
    c B-PER O
    c I-PER B-PER
    c I-PER I-PER
    c I-PER O
    . O O
    . O O
    . B-PER O
    . I-PER O
    d I-PER O
    d I-PER B-PER
    d I-PER I-PER
    d O I-PER 
    . O I-PER
    . O I-PER
    . O O
    . O O
    e B-PER O
    e I-PER O
    e I-PER O
    e I-PER O
    . O O
    . O O
    . O O
    . O O
    f B-PER B-PER
    f I-PER I-PER
    f I-PER I-PER
    f I-PER I-PER
    . O O
    . O O
    . O B-PER
    . O I-PER
    g B-PER I-PER
    g I-PER I-PER
    g I-PER I-PER
    g I-PER I-PER    
    . O I-PER
    . O I-PER
    . O O
    . O O
    h B-PER B-PER
    h I-PER I-PER
    h I-PER I-PER
    h I-PER I-PER  
    . O O
    . O O
    . O O
    . O O
    j B-PER O
    j I-PER O
    j I-PER O
    j I-PER O
    """.strip().split('\n'))

    def test_exact_strictness(self):
        counts = conlleval.evaluate(self.input_file)
        all_metrics, by_type_metrics = conlleval.metrics(counts)        

        self.assertEqual(all_metrics.tp, 3)
        self.assertEqual(all_metrics.fp, 4)
        self.assertEqual(all_metrics.fn, 5)

        self.assertEqual(all_metrics.prec, 3 / 7)
        self.assertEqual(all_metrics.rec, 3 / 8)
        self.assertEqual(all_metrics.fscore, 2 * (3 / 7 * 3 / 8) / (3 / 7 + 3 / 8))

    def test_overlapping_strictness(self):
        counts = conlleval.evaluate(self.input_file)
        all_metrics, by_type_metrics = conlleval.metrics(counts)

        self.assertEqual(all_metrics.tp, 3)  # the 3 exact matches
        self.assertEqual(all_metrics.fp - all_metrics.fp_ov, 1)  # the 1 spurious
        self.assertEqual(all_metrics.fn - all_metrics.fn_ov, 2)  # the 2 missing
        self.assertEqual(all_metrics.fp_ov, 3)  # the 3 overlapping
        self.assertEqual(all_metrics.fn_ov, 3)  # the 3 overlapping        

        self.assertEqual(all_metrics.prec_ov, 9 / 10)
        self.assertEqual(all_metrics.rec_ov, 9 / 11)
        self.assertAlmostEqual(all_metrics.fscore_ov, 2 * (9 / 10 * 9 / 11) / (9 / 10 + 9 / 11), places=5)

    def test_half_overlapping_strictness(self):
        counts = conlleval.evaluate(self.input_file)
        all_metrics, by_type_metrics = conlleval.metrics(counts)

        self.assertEqual(all_metrics.tp, 3)  # the 3 exact matches
        self.assertEqual(all_metrics.fp - all_metrics.fp_ov, 1)  # the 1 spurious
        self.assertEqual(all_metrics.fn - all_metrics.fn_ov, 2)  # the 2 missing
        self.assertEqual(all_metrics.fp_ov, 3)  # the 3 overlapping
        self.assertEqual(all_metrics.fn_ov, 3)  # the 3 overlapping

        self.assertEqual(all_metrics.prec_half_ov, (3 + 6 / 2) / 10)
        self.assertEqual(all_metrics.rec_half_ov, (3 + 6 / 2) / 11)
        self.assertEqual(all_metrics.fscore_half_ov, 2 * ((3 + 6 / 2) / 10 * (3 + 6 / 2) / 11) / ((3 + 6 / 2) / 10 + (3 + 6 / 2) / 11))