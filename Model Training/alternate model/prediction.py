import torch
from torch import nn
import math
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("-"*30)

from torch import nn
import math

from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp((torch.arange(0, embed_dim, 2)) * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1, vocab_size=10000, max_len=5000):
        super(Transformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.y_alt_out = nn.Linear(embed_dim, 4)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.position_encoding(x)

        x = self.encoder(x)
        x = x.mean(dim=1)

        y_alt_out = self.y_alt_out(x)

        return (
            y_alt_out
        )
    
with open("vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

def get_codon(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def get_tensor(text):
    return [vocab.get(codons.lower(), vocab['<UNK>']) for codons in get_codon(text)]

model = Transformer(
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    ff_dim=2048,
    dropout=0.2,
    vocab_size=len(vocab),
    max_len=199
)

checkpoint = torch.load("model_Second Run_epoch_20.pth", map_location=device, weights_only=False) 

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def predict(input_val):
    with torch.no_grad():
        index_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        input = torch.tensor(get_tensor(input_val), dtype=torch.long)

        y_alt = model(input.unsqueeze(dim=0))
        prediction = torch.softmax(y_alt, dim=1)
        output = index_to_base.get(prediction.argmax(dim=1).item(), "Not find")

    print({
        "Predicted Base": output,
        "Confidence": f"{prediction[0][prediction.argmax(dim=1)[0]]*100:.2f}%"
    })

sequence = ['AATACCTGTGAGATGGATACTTTTCCTTTATGGAAGAGATAAGCATTTTTTCAACTTTTTGGTCCGTAGCAGTTACCAAATCAACTGGAGAACTTTTCAGCATAACATTCATTTCATTTTTTATAGCTTCACAAACTACCTAAAAAGAGGTTTTGGTAAGCAAATAGAAAAGGCATAATTTTAAACAAAAAGATAAAATAC',
       'TATCATGCTGCCAGTTACTTCTTCTCCTTCTCGTCTTTGCAGACAGCTGCGAACAGGCTTTAGAGAACCCTGAAGGAGAGCTATGCCTTTTGATTTCTCAGAGTTCTTACAAATTAAGATACTCTCGCCTATAACCAAAAGATACACATTTGCAAAATCCCAATATAGTTCCATAAGATATCTGCTAAAGAGATTTTAAGT',
       'CGCTGGCTGGGGGATGGGTTCTCCTAGGAAATAGCCCTCGTTGTCAGACTCTGAAGAGGAGGAGCAGGTGGAACACCAATCATACTCGGCGAAGTAGGGTCCCCAGCGGTCCCCAAAGGCATTCTGCAAAGCCAGGTCCGACACAGTCCTAGGGCACTGGCCGTACAGGTCCCTCCGGGACCCATGCCCCATGCTCTCCTG',
       'TAAAGACTTCAAGGATTTCCTTGTCAATAGACACATTTTTTAACCTTCCCTTTTAACTGATACATTAGAAAGCCTTATCTTTCAACGGACATACATTGCGAATATAAGAAATTCCAAGTTCAAATAATATTCCAAGATCTGGAGATGCAGAATTAGACCTGATAAAGCAATCACCATCAAGCAAACATGGTAAGATGCGAG',
       'TATAGTATAAAGTCATTAACAAGAAACAGGATATGCTTTAAGACAGAATTCACTGTCTGTTGCTTCAGTAAAAGGACCTCGGGGAATAAAACATTTCTCTCTTATATGCCAGAATGTAGGCTGGTCCCTATGTCATGTCTTCCATTAAGAACACTAAAAAGTCCTTGCAAGAATGGAGATATGCATTCAAGAGAGGTGCTA',
       'GGCACAGTCACTGATGAGACCTTCTTAGAAAAGCTGAACCAAGTATGTGCCACCCACCAGCATTTTGAGAGCAGGATGAGCAAGTGCTCTCGGTTCCTCAATGACACGTCTCTGCCTCACAGCTGCTTCAGGATCCAGCATTATGCTGGAAAGGTATGGGGGAGCTGTGAGCACCCAGTCAGGCCTGCCAAGTTACCCCTG',
       'AGGTGGGGCCTCACAAAAGGGATGGAGGACAGACTGGGCCATCAGAGAATGTTAGGTGGGCAGACTGGACACCTACGATCCTGTCTTTGAGAGAAACGAGCTCCTCCTCCTCTTTCTTCCTGTTCTCAAAGTGAGCCTCGATCAGCGCCTGCAACTCATTCAGGTCCTTCTCCATGCGCTTCCGGTGGATGTCCTGTGGGT',
       'GGGAACTCCAGAGAGGGAGTCCATCTGTTCTCCTTCTCCTCCCCTACCCCAAAACAGCCCCTGGCTCACCTTGGCATGGAGCTTCGTGATGTCAGGTCTCCGCTCAGAGAGGTTGAACAGCTAGAAGGAGCAAAAGGAAGAGAGCATGAGGAGGTGAGGGTGAGGAGGGCGGCGAGGAGAGAGAGAACCATGGCCTGCTGG',
       'CCCCTGCAGGTGCCCAGGGGCTACAACTACAGGGCAGAGGTGAGGAAGCTCATTCCCCAGCTGCAGGTCCTGGACGAAGTGCCGGCCGCACACACAGGCCCACCGGCCCCCCCGCGGCTGAGCCAGGACTGGCTTGCGGTGAAGGAGGCCATCAAGAAGGGCAACGGCCTTCCCCCGCTGGGTACGGCAGCTGCGCCCGGA',
       'TTTTAAAAGTACATAACTTTGAGAAACTTACCACATTCATGTTTGTTTCAGGATCCACAGCTGTCAGTTTCCCTCCAGACCCCTCAGAATGCAGGGTCCAGAGAACCAAACAGACCACCAACCCCAAGAACCGCATTTTCATTCTGTATAATAAAACAGTCTATTATTATTTGCTTGAATTTTACTACACAAAAATATTCT']

for sq in sequence:
    predict(sq)