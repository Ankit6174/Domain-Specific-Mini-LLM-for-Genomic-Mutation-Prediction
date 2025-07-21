# Run this module using WSL terminal download these libraries on that environment and then run this module.

from cyvcf2 import VCF  # type: ignore 
from Bio import SeqIO  # type: ignore
import pandas as pd
from tqdm import tqdm
import os

MAX_ROWS_PER_BATCH = 50000
batch_data = []
batch_num = 0

os.makedirs("output", exist_ok=True)

clnsig_to_label = {
    "Benign": 0,
    "Likely_benign": 1,
    "Uncertain_significance": 2,
    "Likely_pathogenic": 3,
    "Pathogenic": 4,
    "Conflicting_interpretations_of_pathogenicity": 5
}

genome = SeqIO.to_dict(SeqIO.parse("data/GRCh38.fa", "fasta"))
print("Loaded chromosomes:", list(genome.keys())[:10])

vcf = VCF("data/new_clinvar.vcf")

def save_batch(batch_data, batch_num):
    df_batch = pd.DataFrame(batch_data)
    file_path = f"output/clinvar_enhanced_batch_{batch_num}.csv"
    df_batch.to_csv(file_path, index=False)
    print(f"Saved batch {batch_num} with {len(df_batch)} rows → {file_path}")

for variant in tqdm(vcf, desc="Processing variants", unit="variant", ncols=100):
    if not variant.is_snp:
        continue

    chrom = str(variant.CHROM)
    if chrom not in genome:
        chrom = "chr" + chrom
    if chrom not in genome:
        continue

    pos = variant.POS
    ref = variant.REF
    alt = variant.ALT[0]

    raw_clnsig = variant.INFO.get("CLNSIG", None)
    if raw_clnsig is None:
        continue

    clnsig_values = raw_clnsig if isinstance(raw_clnsig, list) else [str(raw_clnsig)]
    label = next((clnsig_to_label[val] for val in clnsig_values if val in clnsig_to_label), None)
    if label is None:
        continue

    try:
        seq = genome[chrom].seq
        start = pos - 101
        end = pos + 100
        if start < 0 or end > len(seq):
            continue

        full_context = str(seq[start:end])
        left_context = str(seq[start:pos-1])
        right_context = str(seq[pos:end])
        
        batch_data.append({
            "sequence": full_context,
            "label": label,
            "mutation_pos": 101,
            "ref": ref,
            "alt": alt,
            "mutation_type": f"{ref}->{alt}",
            "chrom": chrom,
            "genomic_pos": pos,
            "context_left": left_context,
            "context_right": right_context
        })

        if len(batch_data) >= MAX_ROWS_PER_BATCH:
            save_batch(batch_data, batch_num)
            batch_data = []
            batch_num += 1

    except Exception as e:
        print(f"Error at {chrom}:{pos} — {str(e)}")
        continue

if batch_data:
    save_batch(batch_data, batch_num)