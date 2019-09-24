#-*- coding: utf-8 -*-

import os
import numpy as np
import librosa
import argparse

# import torch

def check_and_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_audio_file(file_path, sampling_rate = 16000):
    audio_file = librosa.core.load(file_path, sr=sampling_rate)[0]
    return audio_file

def save_audio_file(file_path, data, sampling_rate = 16000):
    librosa.output.write_wav(file_path, data, sr=sampling_rate)

def noise_injection(data, noise_factor):
    noise = np.random.randn(len(data))
    white_noise = noise_factor * noise
    noised_data = data + white_noise
    noised_data = noised_data.astype(type(data[0]))
    return noised_data

def load_label_list(label_path):
    with open(label_path) as reader:
        label_bulk = reader.read()
    label_list = label_bulk.split("\n")
    if label_list[-1] == '':
        label_list = label_list[0:-1]
    file_names = []
    labels = []
    for label_item in label_list:
        file_name, label = label_item.split(",")
        file_names.append(file_name)
        labels.append(label)
    return file_names, labels

def scan_and_manipulate(in_path, in_label_path, out_path, out_label_path, noise_factor):
    file_names, labels = load_label_list(in_label_path)
    data_list = os.listdir(in_path)
    data_list.sort()
    data_list = data_list[1:]
    for idx, data in enumerate(data_list):
        if file_names[idx] != data.split(".")[0]:
            print("file_name : " + str(file_names[idx]) + " data : " + str(data))
            print("error : data mismatched!")
            return
        raw_audio = load_audio_file(os.path.join(in_path, data))
        manipulated_data = noise_injection(raw_audio, noise_factor)
        save_audio_file(os.path.join(out_path, data), manipulated_data, sampling_rate=16000)

    with open(in_label_path) as reader:
        label_info = reader.read()
        writer = open(out_label_path, "w")
        writer.write(label_info)

def main():

    parser = argparse.ArgumentParser(description='Augmenting speech data')
    parser.add_argument('--sampling_rate', type=int, default=16000, help="sample rate of data (default: 16000)")
    parser.add_argument('--data_path', type=str, default='./sample_dataset', help="data path to manipulate (default: ./sample_dataset)")
    parser.add_argument('--output_path', type=str, default='./manipulated_dataset', help="output path that are manipulated (default: ./manipulated_dataset)")
    parser.add_argument('--noise_factor', type=int, default=0.005, help="noise factor to give white noise for the data (default: 0.005)")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("data_path does not exist : " + args.data_path)
        return

    data_path = os.path.join(args.data_path, "train/train_data")
    label_path = os.path.join(args.data_path, "train/train_label")
    output_path = os.path.join(args.output_path, "train/train_data")
    out_label_path = os.path.join(args.output_path, "train/train_label")

    # csv file copy
    data_list_csv = os.path.join(data_path, "data_list.csv")
    out_data_list_csv = os.path.join(output_path, "data_list.csv")
    with open(data_list_csv, 'r') as reader:
        with open(out_data_list_csv, 'w') as writer:
            writer.write(reader.read())

    check_and_makedir(output_path)

    scan_and_manipulate(data_path, label_path, output_path, out_label_path, noise_factor=args.noise_factor)




if __name__ == "__main__":
    main()
