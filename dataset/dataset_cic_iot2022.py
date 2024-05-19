import os
from tqdm import tqdm
import subprocess
from dataset_common import find_files

def split_pcap_files():
    filenames = find_files("/mnt/ssd1/ciciot2022-raw")
    for filename in tqdm(filenames):
        splitter = "/root/ShieldGPT/pcap_tool/splitter"
        dir_name = os.path.join("/root/NetMamba/dataset/CICIoT2022", "flows", *filename.split("/")[4:-1])
        os.makedirs(dir_name, exist_ok=True)
        flow_prefix = os.path.basename(filename)[:-len(".pcap")]
        with open(os.path.join(dir_name, f"{flow_prefix}.log"), "w") as f:
            subprocess.run(f"{splitter} -i '{filename}' -o '{dir_name}' -p {flow_prefix}- -f five_tuple",
                        shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    split_pcap_files()