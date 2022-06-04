import random
import io
import sys
import os
from tqdm import tqdm
import csv
import glob
from time import process_time


def count_lines(f):
    n_lines = 0
    with open(f, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for _ in csv_reader:
            n_lines = n_lines + 1
    return n_lines


def split(filename, filehandler, delimiter=',',
          output_name_template='output_%s.csv', output_path='./split/', keep_headers=False):
    row_limit = 4999500
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)


def shuffle(file_in, file_out='out'):
    file_out = './out/' + file_out
    files_out = []

    NUM_OF_FILES = 1_00

    for i in range(NUM_OF_FILES):
        f_ = file_out + str(i) + '.csv'
        files_out.append(io.open(f_, 'w', encoding='utf-8'))

    with io.open(file_in, 'r', encoding='utf-8') as source:
        for f in tqdm(source):
            files_out[random.randint(0, NUM_OF_FILES - 1)].write(f)
        for i in range(NUM_OF_FILES):
            files_out[i].close()

    for i in range(NUM_OF_FILES):
        f_ = file_out + str(i) + '.csv'
        data = []
        with io.open(f_, 'r', encoding='utf-8') as file:
            data = [(random.random(), line) for line in tqdm(file)]
        data.sort()
        with io.open(f_, 'w', encoding='utf-8') as file:
            for _, line in tqdm(data):
                file.write(line)


def merge(folder, out_file):
    extension = 'csv'
    all_filenames = [i for i in glob.glob('./'+folder+'/*.{}'.format(extension))]
    for file in all_filenames:
        start = process_time()
        print(file)
        with open(file, 'r') as f:
            f_csv = csv.reader(f)
            with open(out_file, 'a', newline='') as out:
                writer = csv.writer(out)
                for row in f_csv:
                    if row != []:
                        writer.writerow(row)
        print(process_time() - start)


def merge_train_test_validation(folder, train=0.7, test=0.15, val=0.15):
    extension = 'csv'
    all_filenames = [i for i in glob.glob('./'+folder+'/*.{}'.format(extension))]
    train_size = len(all_filenames) * train
    test_size = train_size + len(all_filenames) * test
    count = 0
    for file in all_filenames:
        start = process_time()
        print(file)
        with open(file, 'r') as f:
            f_csv = csv.reader(f)
            if count < train_size:
                out_file = './dataset/train_set.csv'
            elif count < test_size:
                out_file = './dataset/test_set.csv'
            else:
                out_file = './dataset/validation_set.csv'
            with open(out_file, 'a', newline='') as out:
                writer = csv.writer(out)
                for row in f_csv:
                    if row:
                        writer.writerow(row)
        print(process_time() - start)
        count = count + 1

def main():
    args = sys.argv
    f_in = args[1]

    shuffle(f_in)

    os.remove(f_in)

    merge_train_test_validation('out')

    files = glob.glob('./out/*.csv')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    os.rmdir('out')

if __name__ == "__main__":
    main()