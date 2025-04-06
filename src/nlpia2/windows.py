# read windows tsv file
from tqdm import tqdm
import pandas as pd

from nlpia2.string_normalizers import try_decode


if __name__ == '__main__':
    filepath = '/home/hobs/Downloads/10-million-combos/10-million-combos.txt'
    pairs = []
    with open(filepath, 'rb') as stream:
        for i, line in tqdm(enumerate(stream.readlines())):
            rawline = line
            line = line[:-2]  # remove win newline
            if line and b'\t' in line:
                pos = line.index(b'\t')
                pairs.append([line[:pos], line[pos + 1:]])
            else:
                # 5851256 b'markcgilberteternity2969\r\n'
                # 7945278 b'sailer1216soccer1216\r\n'
                print(i, rawline)
                pairs.append([b'', line])

    pairs_decoded = []
    for un, pw in tqdm(pairs):
        pw_decoded = try_decode(pw)
        un_decoded = try_decode(un)
        # reverse the pair order
        pairs_decoded.append([un_decoded, pw_decoded])

    df = pd.DataFrame(pairs_decoded, columns='username password'.split())
    df.to_csv('username_password.csv.gz', compression='gzip', index=False)


def read_tsv(filepath='/home/hobs/Downloads/10-million-combos/10-million-combos.txt'):
    pairs = []
    with open(filepath, 'rb') as stream:
        for line in stream:
            line = line[:-2]  # remove win newling
            if line and b'\t' in line:
                pos = line.index(b'\t')
                pairs.append([line[:pos], line[pos + 1:]])
            else:
                pairs.append([b'', b''])
    return pairs
