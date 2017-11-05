#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re

from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


"""
from nalaf.evaluators
strictness:
Determines whether a text spans matches and how we count that match, 3 possible values:
    * 'exact' count as:
        1 ONLY when we have exact match: (startA = startB and endA = endB)
    * 'overlapping' count as:
        1 when we have exact match
        1 when we have overlapping match
    * 'half_overlapping' count as:
        1 when we have exact match
        0.5 when we have overlapping match
"""

Metrics = namedtuple('Metrics', ['tp', 'fp', 'fn',
                                 'fp_ov', 'fn_ov',
                                 'prec', 'rec', 'fscore',
                                 'prec_ov', 'rec_ov', 'fscore_ov',
                                 'prec_half_ov', 'rec_half_ov', 'fscore_half_ov'])


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.overlapping_chunk = 0
        self.correct_tags = 0  # number of correct chunk tags
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_overlapping_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


class Entity:
    def __init__(self, type, offset, end_offset, text):
        self.type = type
        self.offset = offset
        self.end_offset = end_offset
        self.text = text


def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])  # use defaults

    counts = EvalCounts()

    num_features = None  # number of features per line
    in_correct_matching = False  # currently processed chunks is correct until now
    in_correct = False
    in_guessed = False
    last_correct = 'O'  # previous chunk tag in corpus
    last_correct_type = ''  # type of previously identified chunk tag
    last_guessed = 'O'  # previously identified chunk tag
    last_guessed_type = ''  # type of previous chunk tag in corpus
    last_line = ''

    for line in iterable:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if start_correct:
            in_correct = True

        if start_guessed:
            in_guessed = True

        if (end_correct and in_guessed and not end_guessed
            and last_guessed_type == last_correct_type):
            counts.overlapping_chunk += 1
            counts.t_overlapping_chunk[last_correct_type] += 1

        if (end_guessed and in_correct and not end_correct
            and last_guessed_type == last_correct_type):
            counts.overlapping_chunk += 1
            counts.t_overlapping_chunk[last_correct_type] += 1

        if end_correct:
            in_correct = False

        if end_guessed:
            in_guessed = False

        if in_correct_matching:
            if (end_correct and end_guessed and
                        last_guessed_type == last_correct_type):
                in_correct_matching = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
                # counts.overlapping_chunk += 1
                # counts.t_overlapping_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct_matching = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct_matching = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

        last_line = line

    if in_correct_matching:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total, overlapping):
    tp, fp, fn = correct, guessed - correct, total - correct

    fp_ov = overlapping
    fn_ov = overlapping
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f = safe_div(2 * p * r, p + r)

    _fp_ov = fp - fp_ov
    _fn_ov = fn - fn_ov
    _tp_ov = tp + fp_ov + fn_ov
    p_ov = safe_div(_tp_ov, _tp_ov + _fp_ov)
    r_ov = safe_div(_tp_ov, _tp_ov + _fn_ov)
    f_ov = safe_div(2 * p_ov * r_ov, p_ov + r_ov)

    _tp_half_ov = tp + (fp_ov + fn_ov) / 2
    _fp_half_ov = fp - fp_ov
    _fn_half_ov = fn - fn_ov
    p_half_ov = safe_div(_tp_half_ov, tp + fp_ov + fn_ov + _fp_half_ov)
    r_half_ov = safe_div(_tp_half_ov, tp + fp_ov + fn_ov + _fn_half_ov)
    f_half_ov = safe_div(2 * p_half_ov * r_half_ov, p_half_ov + r_half_ov)

    return Metrics(tp, fp, fn, fp_ov, fn_ov, p, r, f, p_ov, r_ov, f_ov, p_half_ov, r_half_ov, f_half_ov)


# taken from nalaf.evaluators
def safe_div(nominator, denominator):
    try:
        return nominator / denominator
    except ZeroDivisionError:
        return 0.0  # arbitrary; or float('NaN')


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct, c.overlapping_chunk
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t], c.t_overlapping_chunk[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d; overlapping: %d.\n' %
              (c.found_guessed, c.correct_chunk, c.overlapping_chunk))

    out.write('Strict evaluation:\n')
    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100. * c.correct_tags / c.token_counter))
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec))
        out.write('recall: %6.2f%%; ' % (100. * m.rec))
        out.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_found_guessed[i]))

    out.write('Overlapping evaluation:\n')
    if c.token_counter > 0:
        out.write('%27s: %6.2f%%; ' % ('precision', 100. * overall.prec_ov))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec_ov))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore_ov))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec_ov))
        out.write('recall: %6.2f%%; ' % (100. * m.rec_ov))
        out.write('FB1: %6.2f\n' % (100. * m.fscore_ov))

    out.write('Half overlapping evaluation:\n')
    if c.token_counter > 0:
        out.write('%27s: %6.2f%%; ' % ('precision', 100. * overall.prec_ov))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec_half_ov))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore_half_ov))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec_half_ov))
        out.write('recall: %6.2f%%; ' % (100. * m.rec_half_ov))
        out.write('FB1: %6.2f\n' % (100. * m.fscore_half_ov))


def parse_output(counts):
    overall, by_type = metrics(counts)

    # {'all': {'precision': 0., 'f1': 0., 'recall': 0., 'accuracy': 0., 'support': 100}, 'LOC': {...}...}
    conll_parsed_output = {'all': {
        'precision': overall.prec,
        'f1': overall.fscore,
        'recall': overall.rec,
        'accuracy': 100 * counts.correct_tags / counts.token_counter,
        'support': counts.found_guessed,
        'overlapping': {
            'precision': overall.prec_ov,
            'recall': overall.rec_ov,
            'f1': overall.fscore_ov
        },
        'half_overlapping': {
            'precision': overall.prec_half_ov,
            'recall': overall.rec_half_ov,
            'f1': overall.fscore_half_ov
        }
    }}
    for type, scores in by_type.items():
        conll_parsed_output[type] = {
            'precision': scores.prec,
            'f1': scores.fscore,
            'recall': scores.rec,
            'support': counts.t_found_guessed[type],
            'overlapping': {
                'precision': scores.prec_ov,
                'recall': scores.rec_ov,
                'f1': scores.fscore_ov
            },
            'half_overlapping': {
                'precision': scores.prec_half_ov,
                'recall': scores.rec_half_ov,
                'f1': scores.fscore_half_ov
            }
        }
    return conll_parsed_output


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
