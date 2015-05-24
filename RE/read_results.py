import numpy as np

ll = []
with open('log_results_1.txt', 'r') as f:
    for l in f:
        ll.append(float(l.split(':')[2].split(',')[0]))

ll = np.array(ll)
print np.max(ll), np.argmax(ll)

with open('log_params_1.txt', 'r') as f:
    print f.readlines()[np.argmax(ll)]
with open('log_results_1.txt', 'r') as f:
    print f.readlines()[np.argmax(ll)]
