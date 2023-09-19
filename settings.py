chars = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з',
         'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
         'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ',
         'ъ', 'ы', 'ь', 'э', 'ю', 'я']

vowels = ['а', 'е', 'ё', 'и', 'о', "у", "ы", "э", "ю", "я", "^"]
char2id = {
    ch: idx + 1 for idx, ch in enumerate(chars)
}
pairs = [i + j for i, in chars for j in chars] + chars
pair2id = {
    ch: idx + 1 for idx, ch in enumerate(pairs)
}
MAX_WORD_LEN = 35
MAX_ITEMS = 50
BATCH_SIZE = 128
VOCAB_SIZE = len(chars) + 1
HIDDEN_SIZE = 64
MAX_VOWELS = 15
RANDOM_STATE = 42
