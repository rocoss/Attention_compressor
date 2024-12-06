import torch
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib
import torch.nn as nn
import torch.nn.functional as F
from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Размер батча для обучения
batch_size = 64
# Длина последовательности для обучения
seq_length = 15
# Количество единиц в каждом слое GRU
hidden_size = 1024
# Количество слоев GRU
num_layers = 2
# Размер слоя эмбеддинга
embed_size = 512
# Начальная скорость обучения для оптимизатора
learning_rate = 0.0005
#Результат с такими гиперпараметрами
# 100.00%	cross entropy: 2.78	time: 59.59
# Original size: 100000 bytes
# Compressed size: 34809 bytes
# Compression ratio: 2.8728202476371054
# Режим программы, "compress" или "decompress" или "both"
mode = 'both'

path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


class LSTM_attCompress(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        # Создаем слой эмбеддинга для каждого токена в словаре
        # vocab_size: размер словаря (количество уникальных токенов)
        # embed_size: размерность векторов эмбеддинга
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Инициализируем двунаправленный LSTM с указанными параметрами
        # embed_size: размер входного вектора (эмбеддинг)
        # hidden_size: количество единиц в скрытом состоянии LSTM
        # num_layers: количество слоев LSTM
        # batch_first=True: входные данные имеют форму (batch_size, seq_length, embed_dim)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)

        # Слой линейной трансформации для вычисления весов внимания
        # hidden_size * 2: так как LSTM двунаправленный, выход имеет размерность в 2 раза больше
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Слой линейной трансформации для вычисления логарифмической вероятности
        # hidden_size * 2: размер выходного представления после применения внимания
        # vocab_size: количество классов (размер словаря)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # Преобразуем входные данные в векторы эмбеддинга
        # x: входные данные (индексы токенов), форма (batch_size, seq_length)
        embeds = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        # Выполняем передачу через LSTM и получаем результат
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_length, hidden_size * 2)

        # Вычисляем веса внимания для выходов LSTM
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_length, 1)

        # Применяем softmax для нормализации весов внимания по временной дименсии
        attn_weights = F.softmax(attn_weights, dim=1)

        # Применяем веса внимания к выходам LSTM с помощью бродкастирования
        weighted_hn = lstm_out * attn_weights  # (batch_size, seq_length, hidden_size * 2)

        # Суммируем по временной дименсии для получения единого представления
        hn = weighted_hn.sum(dim=1)  # (batch_size, hidden_size * 2)

        # Преобразуем последнее представление в логарифмические вероятности
        logits = self.fc(hn)  # (batch_size, vocab_size)

        # Возвращаем логарифмические вероятности для каждого класса (токена)
        return F.log_softmax(logits, dim=-1)

def get_symbol(index, length, freq, coder, compress, data):
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data):
    loss = 0
    cross_entropy = 0
    denom = 0
    split = math.ceil(length / batch_size)

    model.train()  # Устанавливаем режим тренировки
    optimizer.zero_grad()  # Обнуляем градиенты

    seq_input = seq_input.to(device)  # Перевод на GPU/CPU

    logits = model(seq_input)  # Получаем логиты (batch_size, vocab_size)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    symbols = []
    mask = []

    # Актуализируем вероятности и маски
    for i in range(batch_size):
        freq = np.cumsum(probs[i] * 10000000 + 1)
        index = pos + i * split
        symbol = get_symbol(index, length, freq, coder, compress, data)
        symbols.append(symbol)

        if index < length:
            prob = probs[i][symbol]
            if prob <= 0:
                prob = 1e-6  # Избегаем ошибки с log
            cross_entropy += math.log2(prob)
            denom += 1
            mask.append(1.0)
        else:
            mask.append(0.0)

    # Преобразование символов в one-hot вектор
    symbols = torch.tensor(symbols, device=device)
    input_one_hot = torch.nn.functional.one_hot(symbols, vocab_size).float()

    # Loss calculation
    loss = torch.nn.functional.cross_entropy(logits, input_one_hot, reduction='none')
    loss = loss * torch.tensor(mask, device=device).unsqueeze(1)  # Применяем маску
    loss = loss.mean()  # Среднее значение лосса

    # Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 4)  # Ограничиваем градиенты
    optimizer.step()

    # Обновляем входную последовательность
    seq_input = torch.cat([seq_input[:, 1:], symbols.unsqueeze(1)], dim=1)

    return seq_input, cross_entropy, denom


def process(compress, length, vocab_size, coder, data):
    start = time.time()

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    # Создание модели
    model = LSTM_attCompress(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.eval()  # Устанавливаем режим оценки

    # Инициализация оптимизатора и лосса
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Подготовка первого батча символов
    freq = np.cumsum(np.full(vocab_size, 1.0 / vocab_size) * 10000000 + 1)
    symbols = []
    for i in range(batch_size):
        symbols.append(get_symbol(i * (length // batch_size), length, freq, coder, compress, data))

    seq_input = torch.tensor(symbols, device=device).unsqueeze(1).repeat(1, seq_length)

    pos = 0
    cross_entropy = 0
    denom = 0

    split = math.ceil(length / batch_size)
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'

    while pos < split:
        # Тренировочный/инференс шаг
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data)
        cross_entropy += ce
        denom += d
        pos += 1

        if pos % 5 == 0:
            percentage = 100 * pos / split
            print(template.format(percentage, -cross_entropy / denom, time.time() - start))

    if compress:
        coder.finish()

    print(template.format(100, -cross_entropy / length, time.time() - start))


def compression():
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    char2idx = {u: i for i, u in enumerate(vocab)}
    for _, c in enumerate(text):
        int_list.append(char2idx[c])

    vocab_size = math.ceil(vocab_size / 8) * 8
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)

        out.write(length.to_bytes(5, byteorder='big', signed=False))

        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)

def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:

        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)

        output = [0] * length
        bitin = BitInputStream(inp)

        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        vocab_size = math.ceil(vocab_size / 8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)

        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))

def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compression()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file) / os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()

