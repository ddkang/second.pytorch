import argparse
import os
import numpy as np

def write_out(fname, inds):
    inds.sort()
    with open(fname, 'w') as f:
        for ind in inds:
            f.write('%06d' % ind)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int, required=True) # replicate
    args = parser.parse_args()

    NB_TOTAL = 7481

    all_inds = list(range(NB_TOTAL))

    np.random.seed(0)
    np.random.shuffle(all_inds)
    val_inds = all_inds[0:1000]
    test_inds = all_inds[1000:3000]

    remaining_inds = all_inds[3000:]
    np.random.seed(args.rep)
    np.random.shuffle(all_inds)
    orig_inds = remaining_inds[0:500]
    rest_inds = remaining_inds[500:]

    base_dir = 'round0/rep{}/{}'.format(args.rep, 'rand')
    try:
        os.makedirs(base_dir)
    except:
        pass
    for strat in ['omg', 'omg-all', 'least', 'margin', 'entropy']:
        try:
            new_dir = 'round0/rep{}/{}'.format(args.rep, strat)
            os.symlink('rand', new_dir)
        except:
            print('uh oh')
            sys.exit(0)

    write_out(os.path.join(base_dir, 'val.txt'), val_inds)
    write_out(os.path.join(base_dir, 'test.txt'), test_inds)
    write_out(os.path.join(base_dir, 'train.txt'), orig_inds)
    write_out(os.path.join(base_dir, 'unlabeled.txt'), rest_inds)

if __name__ == '__main__':
    main()
