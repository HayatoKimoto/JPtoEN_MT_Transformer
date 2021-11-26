import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch.tensor as Tensor
from torch.nn.init import xavier_uniform_
import math
from tqdm import tqdm
import MeCab
import pickle
import sentencepiece as spm
import sacrebleu
from beam import beam_search_decoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=42):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class MyTransformer(nn.Module):
    def __init__(self, device, max_len, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", source_vocab_length: int = 16000, target_vocab_length: int = 16000) -> None:
        super(MyTransformer, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_length, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model, padding_idx=pad_id)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(d_model, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.device = device


    def forward(self, src, tgt, src_mask = None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None) -> Tensor:


        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src_key_padding_mask != None:
            src_key_padding_mask = src_key_padding_mask.transpose(0,1)
        if tgt_key_padding_mask != None:
            tgt_key_padding_mask = tgt_key_padding_mask.transpose(0,1)

        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)

        # 未来の情報を参照しないようにするためのmask
        size = tgt.size(0)
        #tgt_mask = ~ torch.triu(torch.ones(size, size)==1).transpose(0,1)
        tgt_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.device)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.out(output)
        return output


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def word2id_tensor(model,data):
    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    with open(data) as f:
        lines=f.readlines()

    id_list = []
    for line in lines:
        ids=sp.EncodeAsIds(line[:40])
        ids.insert(0,1)
        ids.append(2)
        id_list.append(torch.tensor(ids[:max_len]))

    packed_inputs= pack_sequence(id_list,enforce_sorted=False)
    padded_packed_inputs,_ = pad_packed_sequence(packed_inputs, batch_first=True,total_length=max_len)

    return padded_packed_inputs
        


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pad_id = 0
bos_id = 1
eos_id = 2

batch_size=256
max_len=42
learning_rate = 5.8864877297790563e-05

#データの読み込み
X_train = word2id_tensor('sentencepiece-ja.model','kftt-data-1.0/data/orig/kyoto-train.ja')
Y_train = word2id_tensor('sentencepiece-en.model','kftt-data-1.0/data/orig/kyoto-train.en')

X_valid = word2id_tensor('sentencepiece-ja.model','kftt-data-1.0/data/orig/kyoto-dev.ja')
Y_valid = word2id_tensor('sentencepiece-en.model','kftt-data-1.0/data/orig/kyoto-dev.en')


X_test = word2id_tensor('sentencepiece-ja.model','split/test_jesc_ja')
Y_test = word2id_tensor('sentencepiece-en.model','split/test_jesc_en')

#データセットの作成
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(X_valid, Y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

V_ja = 16000
V_en = 16000
model = MyTransformer(device=device, max_len=max_len, source_vocab_length=V_ja, target_vocab_length=V_en).to(device)


optim = optim.Adam(model.parameters(), lr=learning_rate ,betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        
        
def test(test_loader,beam_width):
    model.eval()

    pred_questions = []

    beam_width = beam_width
    n_best = 1
    max_dec_steps = 100
    sp = spm.SentencePieceProcessor()
    sp.Load("sentencepiece-en.model")

    with open('split/test_jesc_en') as f:
        lines=f.readlines()


    
    with torch.no_grad():
        total_score=0
        for (src , _ ), references in zip(test_loader,lines):
            # バッチサイズ1でsrcを取り出す
            # 例： tensor([[   10,  3641,    14, 12805,   256,  6920,  2040,  6126,     8,   165, ...]])
            n_best_list = beam_search_decoding(model, src, beam_width, n_best, bos_id, eos_id, pad_id, max_dec_steps, device)
            hypothesis=[]
            for tok in n_best_list[0][1:]:
                if tok == eos_id:break
                hypothesis.append(sp.DecodeIds([tok]))

            hypothesis=' '.join(hypothesis) 
            references = [references.rstrip()]
            bleu=sacrebleu.corpus_bleu(references,hypothesis)
            total_score+=bleu.score
            
    
    
    return total_score/len(test_loader)


model.load_state_dict(torch.load('best_model_2'))

score_list=[]
for i in range(1,100):  
    BLEUscore=test(test_loader,i)
    score_list.append(BLEUscore)
    print(BLEUscore,i)

fig = plt.figure()
x = range(1,100)
y = score_list
 
plt.plot(x,y)
fig.savefig("94.png")
plt.show()
