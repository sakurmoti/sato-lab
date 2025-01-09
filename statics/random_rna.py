import random
import Levenshtein

def random_rna():
    length = 50
    return ''.join(random.choices('ACGU', k=length))

for _ in range(10):
    seq1 = random_rna()
    seq2 = random_rna()
    print(f'{Levenshtein.ratio(seq1, seq2):.2f}')