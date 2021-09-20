import argparse
import sys
from fairseq.models.transformer_lm import TransformerLanguageModel


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, help='the model dir')

args = parser.parse_args()

lm = TransformerLanguageModel.from_pretrained(
    args.model_dir,
    'model.pt',
    tokenizer='moses',
    bpe='fastbpe')

for line in sys.stdin:
    x = lm.tokenize(line)
    x = lm.apply_bpe(x)
    sys.stdout.write(f'{x}\n')
