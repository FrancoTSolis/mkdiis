"""Fetch 3D structures of common drug-like molecules from PubChem into data/organic.json.

    python prep_pubchem.py
"""
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = """aspirin caffeine ibuprofen acetaminophen nicotine dopamine serotonin epinephrine histamine glucose
morphine codeine cocaine lidocaine procaine benzocaine naproxen salicylic_acid vanillin menthol camphor limonene
capsaicin resveratrol quercetin tryptophan tyrosine phenylalanine histidine arginine adenine guanine cytosine
thymine uracil nicotinamide penicillin_G sulfamethoxazole metformin theobromine melatonin atropine quinine
artemisinin ascorbic_acid citric_acid diclofenac estradiol testosterone curcumin""".split()


def fetch(name):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{urllib.parse.quote(name.replace('_', ' '))}/SDF?record_type=3d")
    with urllib.request.urlopen(url, timeout=60) as r:
        sdf = r.read().decode().splitlines()
    n = int(sdf[3][:3])
    atoms = [[ln[31:34].strip(), float(ln[0:10]), float(ln[10:20]), float(ln[20:30])] for ln in sdf[4:4 + n]]
    charge = 0
    for ln in sdf:
        if ln.startswith("M  CHG"):
            f = ln.split()[3:]
            charge += sum(int(f[i + 1]) for i in range(0, len(f), 2))
    return dict(name=name, charge=charge, atoms=atoms)


def main():
    out = []
    for name in NAMES:
        try:
            out.append(fetch(name))
            print(name, len(out[-1]["atoms"]))
        except Exception as e:
            print("skip", name, e)
        time.sleep(0.3)
    with open(os.path.join(HERE, "data", "organic.json"), "w") as f:
        json.dump(out, f)
    print(len(out), "molecules")


if __name__ == "__main__":
    main()
