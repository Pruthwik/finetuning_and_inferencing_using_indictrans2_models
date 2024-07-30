"""Use IndicTrans2 model for finetuning and inferencing."""
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from transformers import AutoTokenizer
from argparse import ArgumentParser


BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=256, return_tensors='pt'
        ).to(DEVICE)
        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False, skip_special_tokens=True)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the source file path')
    parser.add_argument('--model', dest='mod', help='Enter the model folder path')
    parser.add_argument('--output', dest='out', help='Enter the target file path')
    args = parser.parse_args()
    indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-dist-320M"
    ip = IndicProcessor(inference=True)
    indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(args.mod, trust_remote_code=True)
    indic_indic_model.to(DEVICE)
    indic_indic_tokenizer = AutoTokenizer.from_pretrained(indic_indic_ckpt_dir, trust_remote_code=True)
    hi_sents = read_lines_from_file(args.inp)
    print(len(hi_sents))
    src_lang, tgt_lang = "hin_Deva", "ory_Orya"
    or_translations = batch_translate(hi_sents, src_lang, tgt_lang, indic_indic_model, indic_indic_tokenizer, ip)
    write_lines_to_file(or_translations, args.out)
    # flush the models to free the GPU memory
    del indic_indic_tokenizer, indic_indic_model


if __name__ == '__main__':
    main()
