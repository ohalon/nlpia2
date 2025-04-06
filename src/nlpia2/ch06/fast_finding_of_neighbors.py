import random
from annoy import AnnoyIndex
from nessvec.util import load_glove
from tqdm import tqdm


def find_similar(w, n=10):
    return [i2w[i] for i in aidx.get_nns_by_item(w2i['seattle'], 10)]


try:
    w2v['the']
except:
    w2v = load_glove()

class VectorIndex():

    def __init__(self, w2v=w2v, num_trees=10):
        self.idx = AnnoyIndex(len(w2v['the']), 'angular')

        print(f'Building index on {len(w2v)} {len(w2v[w2v.keys()[0]])}-D vectors')
        for i, v in tqdm(enumerate(w2v.values())):
            self.idx.add_item(i, v)

        self.idx.build(num_trees)
        self.i2w = list(w2v.keys())
        self.w2i = dict(zip(i2w, range(len(i2w))))

        return dict(aidx=aidx, i2w=i2w, w2i=w2i)

    def find_similar(w, n=10):
        return [i2w[i] for i in aidx.get_nns_by_item(w2i['seattle'], 10)]


if __name__ == "__main__":
    seattle = w2v['seattle']
    d = create_index()
    aidx = d['aidx']
    aidx.save(f'glove{len(w2v["the"])}.ann')
