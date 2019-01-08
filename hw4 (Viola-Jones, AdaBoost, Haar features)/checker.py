#!/usr/bin/env python2
from __future__ import print_function

import argparse
import os
import subprocess as sp
import logging

CATEGORIES = ['cars', 'faces']

def _parse_args():
    parser = argparse.ArgumentParser(prog='checks homework #4 of bsu ml course')
    parser.add_argument('--show-failed-examples', action='store_true', help='print file names of examples on which algo has failed')
    parser.add_argument('executable', help='path to executable which is implementing the algorithm')
    parser.add_argument('model', help='this file will be passed to executable as the first argument')
    parser.add_argument('set', help='path/to/folder/with/cars/and/faces/subfolder')
    return parser.parse_args()

def _run_algo(executable, model, fname):
    cmd = [executable, model, fname]
    result = sp.check_output(cmd).strip()
    if result == '0':
        return 0
    if result == '1':
        return 1
    raise RuntimeError('Expected 0/1 output, but got "{}"'.format(result))

def _main(args):
    done = 0
    success = 0

    for i, category in enumerate(CATEGORIES):
        folder = os.path.join(args.set, category)
        assert os.path.isdir(folder), "Folder {} doesn't exist".format(folder)
        for fname in os.listdir(folder):
            full_path = os.path.join(folder, fname)
            if not full_path.endswith('.bmp'):
                logging.warning("File {} doesn't have bmp extension".format(full_path))

            try:
                result = _run_algo(args.executable, args.model, full_path)
            except:
                logging.error("Running algo on %s failed", full_path)
                raise

            done += 1
            if result == i:
                success += 1
            else:
                if args.show_failed_examples:
                    print("{}\tneed {}\tgot {}".format(full_path, CATEGORIES[i], CATEGORIES[result]))

    assert done > 0, "Found 0 files to check"
    accuracy = float(success) / done * 100.
    print('You algo got {} answers correct out of {}'.format(success, done))
    print('Accuracy: {:.02f}'.format(accuracy))
    return 0

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    exit(_main(_parse_args()))
