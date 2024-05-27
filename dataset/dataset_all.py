import os
from tqdm import tqdm
from dataset_common import find_files, read_5hp_list
import subprocess
import json
from PIL import Image
import random
import numpy as np
import shutil
import argparse


def sample_pcap(minimum=200, maximum=3000, input_dir="CICIoT2022/flows/1-Power",
                output_dir="CICIoT2022/flows_sampled/1-Power", if_cic=False):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    random.seed(0)
    os.makedirs(output_dir, exist_ok=True)
    sub_dirs = list(filter(lambda x: os.path.isdir(f"{input_dir}/{x}"), os.listdir(input_dir)))
    if if_cic:
        pcap_files = []
        for sub_dir in sub_dirs:
            pcap_files.extend(find_files(f"{input_dir}/{sub_dir}", extension=".pcap"))
        if len(pcap_files) < minimum: # Skip if less than minimum
            print(f"Skip {input_dir} due to less than {minimum} flows")
            return
        if len(pcap_files) > maximum:
            pcap_files = random.sample(pcap_files, maximum)
        for pcap_file in tqdm(pcap_files, desc=output_dir):
            compressed_name = pcap_file[len(input_dir):].replace("/", "_")
            dst_pcap_file = f"{output_dir}/{compressed_name}"
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)
    else:
        for sub_dir in sub_dirs:
            pcap_files = find_files(f"{input_dir}/{sub_dir}", extension=".pcap")
            if len(pcap_files) < minimum: # Skip if less than minimum
                print(f"Skip {input_dir}/{sub_dir} due to less than {minimum} flows")
                continue
            if len(pcap_files) > maximum:
                pcap_files = random.sample(pcap_files, maximum)
            for pcap_file in tqdm(pcap_files, desc=f"{output_dir}/{sub_dir}"):
                dst_pcap_file = pcap_file.replace(input_dir, output_dir)
                os.makedirs("/".join(dst_pcap_file.split("/")[:-1]), exist_ok=True)
                subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)

def sample_all_pcap():
    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Audio", output_dir=f"CICIoT2022/flows_sampled/Audio",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Cameras", output_dir=f"CICIoT2022/flows_sampled/Cameras",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Home Automation", output_dir=f"CICIoT2022/flows_sampled/Home Automation",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/1-Flood", output_dir=f"CICIoT2022/flows_sampled/Flood",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/2-RTSP Brute Force/Hydra", 
                output_dir=f"CICIoT2022/flows_sampled/Hydra", minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/2-RTSP Brute Force/Nmap", 
                output_dir=f"CICIoT2022/flows_sampled/Nmap", minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CrossPlatform/flows/android", output_dir=f"CrossPlatform-Android/flows_sampled",
                    minimum=50, maximum=2000,)
    sample_pcap(input_dir="CrossPlatform/flows/ios", output_dir=f"CrossPlatform-iOS/flows_sampled",
                    minimum=50, maximum=2000,)
    sample_pcap(input_dir="/mnt/ssd1/ISCXVPN2016/flows", output_dir=f"ISCXVPN2016/flows_sampled",
                    minimum=500, maximum=4000,)
    sample_pcap(input_dir="/mnt/ssd1/USTC-TFC2016/flows", output_dir=f"USTC-TFC2016/flows_sampled",
                    minimum=500, maximum=2000,)
    sample_pcap(input_dir="/mnt/ssd1/ISCXTor2016/flows", output_dir=f"ISCXTor2016/flows_sampled",
                    minimum=10, maximum=4000,)

def pcap_to_array(pcap_dir, if_augment=False):
    assert pcap_dir.split("/")[-1] == "flows_sampled"
    image_dir = pcap_dir.replace("flows_sampled", "array_sampled")
    flow_dir_names = os.listdir(pcap_dir)
    for flow_dir_name in flow_dir_names:
        os.makedirs(f"{image_dir}/{flow_dir_name}", exist_ok=True)
        pcap_filenames = os.listdir(f"{pcap_dir}/{flow_dir_name}")
        for pcap_filename in tqdm(pcap_filenames, desc=flow_dir_name):
            try:
                if not if_augment:
                    image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}.png"
                    stat_filename = image_filename.replace(".png", ".json")
                    res = read_5hp_list(f"{pcap_dir}/{flow_dir_name}/{pcap_filename}")[0]
                    flow_array = res.pop("data")
                    image = Image.fromarray(flow_array.reshape(40, 40).astype(np.uint8))
                    image.save(image_filename)
                    with open(stat_filename, "w") as f:
                        json.dump(res, f)
                else:
                    res_list = read_5hp_list(f"{pcap_dir}/{flow_dir_name}/{pcap_filename}", if_augment=True)
                    for i, res in enumerate(res_list):
                        image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}-{i}.png"
                        stat_filename = image_filename.replace(".png", ".json")
                        flow_array = res.pop("data")
                        image = Image.fromarray(flow_array.reshape(40, 40).astype(np.uint8))
                        image.save(image_filename)
                        with open(stat_filename, "w") as f:
                            json.dump(res, f)
            except Exception as e:
                print(f"Error processing {pcap_filename}: {e}")

def all_pcap_to_array():
    pcap_to_array("CICIoT2022/flows_sampled", if_augment=False)
    pcap_to_array("CrossPlatform-Android/flows_sampled", if_augment=True)
    pcap_to_array("CrossPlatform-iOS/flows_sampled", if_augment=True)
    pcap_to_array("ISCXVPN2016/flows_sampled", if_augment=False)
    pcap_to_array("USTC-TFC2016/flows_sampled", if_augment=False)
    pcap_to_array("ISCXTor2016/flows_sampled", if_augment=True)


def split_dataset(input_dir, train_ratio=0.8, valid_ratio=0.1):
    assert input_dir.split("/")[-1] == "array_sampled"
    dir_name = "dataset_sampled"
    output_dir = input_dir.replace("array_sampled", dir_name)
    train_dir = f"{output_dir}/train"
    valid_dir = f"{output_dir}/valid"
    test_dir = f"{output_dir}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(0)
    filenames = find_files(input_dir, extension=".png")
    np.random.shuffle(filenames)
    train_size = int(len(filenames) * train_ratio)
    valid_size = int(len(filenames) * valid_ratio)
    train_files = filenames[:train_size]
    valid_files = filenames[train_size:train_size+valid_size]
    test_files = filenames[train_size+valid_size:]
    for filename in tqdm(filenames, desc="Splitting"):
        if filename in train_files:
            split_type = "train"
        elif filename in valid_files:
            split_type = "valid"
        else:
            split_type = "test"
        label, base_name = filename.split("/")[-2:]
        os.makedirs(f"{output_dir}/{split_type}/{label}", exist_ok=True)
        subprocess.run(f"cp '{filename}' '{output_dir}/{split_type}/{label}/{base_name}'", shell=True)
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

def split_all_datasets():
    datasets = ["CICIoT2022", "CrossPlatform-Android", "CrossPlatform-iOS", "ISCXVPN2016", "USTC-TFC2016", "ISCXTor2016"]
    for dataset in datasets:
        split_dataset(f"{dataset}/array_sampled")

def merge_dataset():
    filenames = []
    datasets = ["CICIoT2022", "CrossPlatform-Android", "CrossPlatform-iOS", "ISCXVPN2016", "USTC-TFC2016", "ISCXTor2016"]
    for dataset in datasets:
        filenames += find_files(f"{dataset}/array_sampled", extension=".png")
    for filename in tqdm(filenames, desc="Merging"):
        filename_list = filename.split("/")
        label = filename_list[0] + "-" + filename_list[-2]
        base_name = filename_list[-1]
        os.makedirs(f"pretrain_dataset/train/{label}", exist_ok=True)
        dst_filename = f"pretrain_dataset/train/{label}/{base_name}"
        subprocess.run(f"cp '{filename}' '{dst_filename}'", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Sample pcap files")
    parser.add_argument("--array", action="store_true", help="Convert pcap files to array")
    parser.add_argument("--split", action="store_true", help="Split dataset into train, valid, and test sets for finetuning")
    parser.add_argument("--merge", action="store_true", help="Merge all datasets into one for pretraining")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.sample or args.all:
        sample_all_pcap()
    if args.array or args.all:
        all_pcap_to_array()
    if args.split or args.all:
        split_all_datasets()
    if args.merge or args.all:
        merge_dataset()
