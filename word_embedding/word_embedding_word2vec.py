import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    filename = os.path.basename(sys.argv[0])
    logger = logging.getLogger(filename)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 4:
        print(globals()['__doc__'])
        sys.exit(1)

    corpus_file, out_model, out_vector = sys.argv[1:4]

    model = Word2Vec(LineSentence(corpus_file), sg=1, vector_size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    model.save(out_model)
    model.wv.save_word2vec_format(out_vector, binary=False)
