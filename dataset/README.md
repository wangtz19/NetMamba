## Data Preprocessing
- Split individual pcap files
For `x` in [`cic_iot2022`, `cross_platform`, `iscx_tor2016`, `iscx_vpn2016`, `ustc_tfc2016`], download corresponding pcap files from their official websites, and then run the following command to execute `Flow Splitting`
```
python dataset_x.py
```

- Porcess all datasets
    - Sample with lower bound and upper bound
    ```
    python dataset_all.py --sample
    ```
    - Execute `Packet Parsing`, `Packet Cropping & Padding`, and `Packet Concatenating`
    ```
    python dataset_all.py --array
    ```
    - Split train, valid, and test sets for fine-tuning
    ```
    python dataset_all.py --split
    ```
    - Merge all datasets for pre-training
    ```
    python dataset_all.py --merge
    ```

## Dependency
- `editcap`: For changing `.pcapng` into `.pcap` formats, see [editcap](https://www.wireshark.org/docs/man-pages/editcap.html).
- `mergecap`: For merging individual pcap files into an integrated one, see [mergecap](https://www.wireshark.org/docs/man-pages/mergecap.html).
- `splitter`: For efficiently splitting traffic flows from raw pcap files, this is a customized tool, see [ShieldGPT/pcap_tool](https://github.com/wangtz19/ShieldGPT/tree/master/pcap_tool)