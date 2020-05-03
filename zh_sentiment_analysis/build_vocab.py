data_path = "data/zh_sentiment_dataset_seg.txt"

file = open(data_path, 'r', encoding='utf-8')

vocab = []

while True:
    line = file.readline()
    if not line:
        break
    label, sentence = line.split('\t')
    for word in sentence.split(" "):
        if word and word not in vocab:
            vocab.append(word)
        else:
            continue

file.close()

with open("vocab_words1.txt", 'w', encoding="utf-8") as file:
    for line in vocab:
        file.write(line + '\n')

# '1\t手机 质量 不错 ， 帮 亲戚 购买 的 。\n'
# \t  分割
