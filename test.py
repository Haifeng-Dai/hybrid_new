import argparse


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--a', type=int, nargs='+', help='test')
    args = parser.parse_args()
    return args


args = get_args()

print(type(args.a))
