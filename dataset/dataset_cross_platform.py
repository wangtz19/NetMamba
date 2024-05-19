import os
from tqdm import tqdm
import subprocess
from dataset_common import find_files

def pcapng_to_pcap():
    # these original files are in pcapng formats, though the extension is .pcap 
    pcap_files = find_files("CrossPlatform/pcap")
    for pcap_file in tqdm(pcap_files):
        subprocess.run(f"editcap -F libpcap {pcap_file} {pcap_file}",
                    shell=True)

def split_pcap_files():
    splitter = "/root/ShieldGPT/pcap_tool/splitter"
    for country in ["china", "india", "us"]:
        for os_type in ["android", "ios"]:
            filenames = os.listdir(f"/root/NetMamba/dataset/CrossPlatform/pcap/{country}/{os_type}")
            for filename in tqdm(filenames):
                app_name = filename[:-len('.pcap')]
                os.makedirs(f"/root/NetMamba/dataset/CrossPlatform/flows/{os_type}/{app_name}",
                            exist_ok=True)
                flow_prefix = f"{country}-{os_type}-{app_name}"
                with open(f"/root/NetMamba/dataset/CrossPlatform/flows/{os_type}/{app_name}.log", "w") as f:
                    subprocess.run(f"{splitter} -i '/root/NetMamba/dataset/CrossPlatform/pcap/{country}/{os_type}/{filename}' -o '/root/NetMamba/dataset/CrossPlatform/flows/{os_type}/{app_name}' -p {flow_prefix}- -f five_tuple",
                                shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    # pcapng_to_pcap()
    split_pcap_files()