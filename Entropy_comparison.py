import os
import math
import random
from collections import Counter
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# --- ENTROPY FUNCTIONS ---

def shannon_entropy(column):
    counts = Counter(res for res in column if res not in ['-', 'X', 'N'])
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def compute_entropy_vector(alignment):
    return [shannon_entropy(alignment[:, i]) for i in range(alignment.get_alignment_length())]

# --- SUBSAMPLING FUNCTIONS ---

def rarefy_alignment(alignment, n_samples):
    return MultipleSeqAlignment(random.sample(list(alignment), n_samples))

def entropy_for_resample(args):
    alignment_list, n_samples = args
    sub_align = MultipleSeqAlignment(random.sample(alignment_list, n_samples))
    return compute_entropy_vector(sub_align)

def bootstrap_entropy_parallel(file_path, n_samples=5000, n_iter=100, n_cores=25):
    print(f"Bootstrapping entropy for: {file_path} using {n_cores} cores")
    alignment = AlignIO.read(file_path, "fasta")
    alignment_list = list(alignment)

    with Pool(processes=n_cores) as pool:
        args = [(alignment_list, n_samples)] * n_iter
        all_runs = pool.map(entropy_for_resample, args)

    mean_entropy = np.mean(all_runs, axis=0)
    std_entropy = np.std(all_runs, axis=0)
    return mean_entropy, std_entropy


# --- Z-SCORE CALCULATION ---

def calculate_z_scores(entropy_matrix):
    mean = np.mean(entropy_matrix, axis=0)
    std = np.std(entropy_matrix, axis=0)
    z_scores = (entropy_matrix - mean) / std
    return z_scores

# --- MAIN EXECUTION ---

alignments = {
    "H1N1": "h1n1_human_aligned.fasta",
    "H3N2": "h3n2_human_aligned.fasta",
    "H5N1": "h5n1_all_2344b_aligned.fasta"
}

raw_entropies = {}
bootstrap_entropies = {}
bootstrap_std = {}
min_seq_count = float('inf')

# 1. Load and calculate raw entropy + find minimum n for bootstrapping
for label, path in alignments.items():
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    alignment = AlignIO.read(path, "fasta")
    raw_entropies[label] = compute_entropy_vector(alignment)
    seq_count = len(alignment)
    min_seq_count = min(min_seq_count, seq_count)
    print(f"{label}: {seq_count} sequences")

# 2. Perform bootstrapped entropy with matched size
for label, path in alignments.items():
    if not os.path.exists(path):
        continue
    mean_entropy, std_dev = bootstrap_entropy_parallel(path, n_samples=min_seq_count, n_iter=100, n_cores=48)
    bootstrap_entropies[label] = mean_entropy
    bootstrap_std[label] = std_dev

# 3. Assemble into DataFrame
max_len = max(len(v) for v in raw_entropies.values())
df = pd.DataFrame({"Position": list(range(1, max_len + 1))})

for label in alignments:
    df[f"Entropy_{label}"] = raw_entropies.get(label, []) + [None] * (max_len - len(raw_entropies.get(label, [])))
    df[f"Bootstrapped_{label}"] = list(bootstrap_entropies.get(label, [])) + [None] * (max_len - len(bootstrap_entropies.get(label, [])))
    df[f"SD_{label}"] = list(bootstrap_std.get(label, [])) + [None] * (max_len - len(bootstrap_std.get(label, [])))

# 4. Z-score entropy
boot_matrix = np.array([bootstrap_entropies[k] for k in alignments if k in bootstrap_entropies])
z_matrix = calculate_z_scores(boot_matrix)

for i, label in enumerate(alignments):
    if label in bootstrap_entropies:
        df[f"Zscore_{label}"] = list(z_matrix[i]) + [None] * (max_len - len(z_matrix[i]))

# 5. Delta entropy comparisons
def delta_entropy(label1, label2):
    e1 = np.array(bootstrap_entropies[label1])
    e2 = np.array(bootstrap_entropies[label2])
    delta = e1 - e2
    return list(delta) + [None] * (max_len - len(delta))

df["Delta_H1N1_vs_H5N1"] = delta_entropy("H1N1", "H5N1")
df["Delta_H3N2_vs_H5N1"] = delta_entropy("H3N2", "H5N1")
df["Delta_H1N1_vs_H3N2"] = delta_entropy("H1N1", "H3N2")

# 6. Save CSV
df.to_csv("entropy_comparison_summary.csv", index=False)
print("Saved entropy_comparison_summary.csv")

# 7. Plotting (basic overview)
plt.figure(figsize=(14, 6))
for label in alignments:
    if label in bootstrap_entropies:
        plt.plot(df["Position"], df[f"Bootstrapped_{label}"], label=label)
plt.xlabel("Site Position")
plt.ylabel("Bootstrapped Shannon Entropy")
plt.title("Subsampled Entropy by Virus Subtype")
plt.legend()
plt.tight_layout()
plt.savefig("entropy_bootstrapped_plot.pdf")
print("Saved entropy_bootstrapped_plot.pdf")
