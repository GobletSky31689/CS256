# Sample Command: python sticky_snippet_generator.py num_snippets mutation_rate from_ends output_file
# CLASS SET: NONSTICK, 12-STICKY, 34-STICKY, 56-STICKY, 78-STICKY, STICK_PALINDROME

import sys
import random


NUM_SNIPPETS = int(sys.argv[1])
MUTATION_RATE = float(sys.argv[2])
FROM_ENDS = int(sys.argv[3])
OUTPUT_FILE = sys.argv[4]
GENE_POOL = ['A', 'B', 'C', 'D']
LEN_GENE_POOL = len(GENE_POOL)


def get_random_sticky_snippet():
    base_snippet = ['']*40
    for ind in xrange(20):
        gene = int(random.uniform(0, LEN_GENE_POOL))
        base_snippet[ind] = GENE_POOL[gene]
        base_snippet[39-ind] = GENE_POOL[(gene+2) % LEN_GENE_POOL]
    return base_snippet


def mutate(x, prob):
    return x if random.random() >= prob else GENE_POOL[int(GENE_POOL.index(x)+random.uniform(1, 4)) % 4]


model_file = open(OUTPUT_FILE, "w")

for _ in xrange(NUM_SNIPPETS):
    source_snippet = get_random_sticky_snippet()
    for i in xrange(20):
        source_snippet[i] = mutate(source_snippet[i], MUTATION_RATE if i < FROM_ENDS else 1)
    model_file.write(''.join(source_snippet)+'\n')

model_file.close()






