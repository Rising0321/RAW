import argparse
from minRAW import (train_model, generate_sentence, export_embeddings, dump_region)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='minRAW',
        description='PyTorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(dest='subcommands')  # , required=True)
    # The above code is modified for compatibility. Argparse in Python 3.6
    # version does not support `required` option in `add_subparsers`.

    train_model.add_subparser(subparsers)
    generate_sentence.add_subparser(subparsers)
    export_embeddings.add_subparser(subparsers)
    dump_region.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
