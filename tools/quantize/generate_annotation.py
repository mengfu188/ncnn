import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to the image folder')
    parser.add_argument('output', default='.', help='path to the output')
    return parser.parse_args()

def main():
    args = get_args()
    contents = []
    labels = []
    for label in os.listdir(args.path):
        labels.append(label)
        parent = os.path.join(args.path, label)
        for file_name in os.listdir(parent):
            file_path = os.path.join(parent, file_name)
            contents.append('{},{}'.format(file_path, label))
    with open(os.path.join(args.output, 'label.txt'), 'w') as f:
        for label in labels:
            f.write(label + '\n')
    with open(os.path.join(args.output, 'annotation.txt'), 'w') as f:
        for content in contents:
            f.write(content + '\n')


if __name__ == "__main__":
    main()