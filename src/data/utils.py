import torch
from torch.utils.data import Dataset

from settings import MAX_WORD_LEN, vowels, char2id, pair2id


class TrainDataset(Dataset):
    def __init__(self, words, tokenizer):
        self.item_list = list(map(tokenizer, words))
        self.labels = list(map(get_item_labels, words))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        word_len = len(self.item_list[idx])
        return ([0] * (MAX_WORD_LEN - word_len) + self.item_list[idx], self.labels[idx])


def train_collate_fn(x):
    item_list, labels = zip(*x)
    item_tensor = torch.tensor(item_list)
    labels_tensor = torch.tensor(labels)
    return {"items": item_tensor, "labels": labels_tensor}


class InferenceDataset(Dataset):
    def __init__(self, words, tokenizer):
        self.item_list = list(map(tokenizer, words))

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return [0] * (MAX_WORD_LEN - len(self.item_list[idx])) + self.item_list[idx]


def inference_collate_fn(x):
    item_list = x
    item_tensor = torch.tensor(item_list)
    return {"items": item_tensor}


def get_item_list(word):
    item = [char2id[ch] for ch in word if ch in char2id]
    return item


def get_pair_list(word):
    pairs = []
    for i in range(0, len(word), 2):
        pair = word.replace("^", "")[i:i + 2]
        if pair in pair2id:
            pairs.append(pair2id[pair])
    return pairs


def get_item_labels(word):
    vow_word = [ch for ch in word if ch in vowels]
    return vow_word.index("^")


def read_words_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            words = file.read().splitlines()
        return words
    except FileNotFoundError:
        print("Файл не найден")
        return []


def new_insert_carot(w, ks):
    vowels = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
    words = []
    for k in ks:
        count = 0
        w1 = ""

        for char in w:
            w1 += char

            if char.lower() in vowels:
                count += 1
                if count == k:
                    w1 += "^"
        words.append(w1)

    for w1 in words:
        if "^" in w1:
            return w1
