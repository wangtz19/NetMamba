import os
from tqdm import tqdm
import subprocess

def merge_pcap_files():
    for category in ["SMB", "Weibo"]:
        dir_name = f"/mnt/ssd1/USTC-TFC2016/Benign/{category}"
        pcap_files = list(filter(lambda x: x.endswith(".pcap"), os.listdir(dir_name)))
        cmd = f"mergecap -w /mnt/ssd1/USTC-TFC2016/Benign/{category}.pcap"
        for pcap_file in pcap_files:
            cmd += f" {dir_name}/{pcap_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)

def split_pcap_files():
    splitter = "/root/ShieldGPT/pcap_tool/splitter"
    for label in ["Benign", "Malware"]:
        filenames = os.listdir(f"/mnt/ssd1/USTC-TFC2016/{label}")
        pcap_file_or_dir_names = list(filter(lambda x: x.endswith(".pcap") or os.path.isdir(x), filenames))
        for name in tqdm(pcap_file_or_dir_names):
            
            if not os.path.isdir(f"/mnt/ssd1/USTC-TFC2016/{label}/{name}"):
                os.makedirs(f"/mnt/ssd1/USTC-TFC2016/flows/{name[:-len('.pcap')]}", exist_ok=True)
                flow_prefix = f"{label}-{name[:-len('.pcap')]}"
                with open(f"/mnt/ssd1/USTC-TFC2016/flows/{name[:-len('.pcap')]}.log", "w") as f:
                    subprocess.run(f"{splitter} -i '/mnt/ssd1/USTC-TFC2016/{label}/{name}' -o '/mnt/ssd1/USTC-TFC2016/flows/{name[:-len('.pcap')]}' -p {flow_prefix}- -f five_tuple",
                                shell=True, stdout=f, stderr=subprocess.STDOUT)
            else:
                filenames = os.listdir(f"/mnt/ssd1/USTC-TFC2016/{label}/{name}")
                for filename in filenames:
                    os.makedirs(f"/mnt/ssd1/USTC-TFC2016/flows/{name}/{filename}", exist_ok=True)
                    flow_prefix = f"{label}-{name}"
                    with open(f"/mnt/ssd1/USTC-TFC2016/flows/{name}.log", "w") as f:
                        subprocess.run(f"{splitter} -i '/mnt/ssd1/USTC-TFC2016/{label}/{name}/{filename}' -o '/mnt/ssd1/USTC-TFC2016/flows/{name}' -p {flow_prefix}- -f five_tuple",
                                    shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    # merge_pcap_files()
    split_pcap_files()