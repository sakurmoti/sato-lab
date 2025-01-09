import time,sys
sys.path.append('../')

from GA.modules import Deg_predictor, TE_predictor

Degpred = Deg_predictor.predictor()
# TEpred = TE_predictor.predictor()

start = time.time()

seq = 'CCGCGCTCTGCAGCCGCAGACCCGGTCCACACGGCCAGGGGCTACGACCCTTGGGATCTGCCCTCCGCTCAGCTCGAGCTTCCCTCGTGGCCGACGGAAC'

f = Deg_predictor.score_sum(Degpred.predict(seq))

end = time.time()
print(f'elapsed_time: {end - start}[sec]')
