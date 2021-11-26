import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    '--input=train_merge_ja, --model_prefix=sentencepiece-ja --character_coverage=0.9995 --vocab_size=16000 --pad_id=0 --unk_id=3'
)


spm.SentencePieceTrainer.Train(
    '--input=train_merge_en, --model_prefix=sentencepiece-en --character_coverage=1.0 --vocab_size=16000 --pad_id=0 --unk_id=3'
)







#日本語
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece-ja.model")

with open('kftt-data-1.0/data/orig/kyoto-train.ja') as f:
    lines=f.readlines()

line=lines[0]
print(line)
#雪舟（せっしゅう、1420年（応永27年）-1506年（永正3年））は号で、15世紀後半室町時代に活躍した水墨画家・禅僧で、画聖とも称えられる。
tokens=line.split()
print(tokens)
#['日本', 'の', '水墨', '画', 'を', '一変', 'さ', 'せ', 'た', '。']
print(sp.DecodePieces(tokens))
#日本の水墨画を一変させた。

ids=sp.EncodeAsIds(line)
print(ids)
#[7, 14274, 9, 604, 624, 2878, 4, 194, 212, 12, 9, 1914, 494, 12, 173, 5393, 53, 12, 9, 3803, 33, 12, 8, 8, 10, 230, 17, 4, 149, 4128, 8803, 3703, 11578, 45, 15, 8335, 17, 4, 542, 734, 213, 10204, 580, 5]
print(sp.DecodeIds(ids))
#雪舟(せっしゅう、1420年(応永27年)-1506年(永正3年))は号で、15世紀後半室町時代に活躍した水墨画家・禅僧で、画聖とも称えられる。
print(sp.GetPieceSize())
#16000

print(sp.IdToPiece(10))
#1


#英語
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece-en.model")

with open('kftt-data-1.0/data/orig/kyoto-train.en') as f:
    lines=f.readlines()

line=lines[0]
print(line)
#Known as Sesshu (1420 - 1506), he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century, and was called a master painter.

tokens=line.split()
print(tokens)
#['Known', 'as', 'Sesshu', '(1420', '-', '1506),', 'he', 'was', 'an', 'ink', 'painter', 'and', 'Zen', 'monk', 'active', 'in', 'the', 'Muromachi', 'period', 'in', 'the', 'latter', 'half', 'of', 'the', '15th', 'century,', 'and', 'was', 'called', 'a', 'master', 'painter.']
print(sp.DecodePieces(tokens))
#HerevolutionizedtheJapaneseinkpainting.
ids=sp.EncodeAsIds(line)
print(ids)
#[11326, 95, 19, 11588, 5291, 1901, 97, 255, 2754, 37, 26, 15, 47, 3264, 3453, 8, 550, 1059, 1622, 10, 4, 474, 51, 10, 4, 1085, 568, 6, 4, 255, 115, 438, 5, 8, 15, 68, 13, 694, 3453, 7]
print(sp.DecodeIds(ids))
#Known as Sesshu (1420 - 1506), he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century, and was called a master painter.
print(sp.GetPieceSize())
#16000
print(sp.IdToPiece(10))