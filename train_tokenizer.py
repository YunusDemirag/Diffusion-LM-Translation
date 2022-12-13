from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

''' This creates a BPE Tokenizer on the covost dataset'''

COVOST_PATH = "../../covost-dataset"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=10000)
tokenizer.pre_tokenizer = Whitespace()
files = [f'{COVOST_PATH}/covost_v2.de_en.{split}.txt' for split in ["test", "train", "dev"]]
tokenizer.train(files, trainer)

tokenizer.save(f"{COVOST_PATH}/tokenizer.json")