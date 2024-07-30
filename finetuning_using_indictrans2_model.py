"""Finetuning the indictrans2 model on your dataset."""
from argparse import ArgumentParser
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from datasets import Dataset
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
from IndicTransTokenizer import IndicDataCollator
from transformers import AutoTokenizer


BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = IndicProcessor(inference=False)


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def create_source_target_pairs(lines):
    """Create source and target pairs from lines."""
    source_sents, target_sents = [], []
    for line in lines:
        src, tgt = line.split('\t')[: 2]
        source_sents.append(src)
        target_sents.append(tgt)
    return source_sents, target_sents


def preprocess_function(sources, targets, tokenizer):
    all_elements = []
    for src_sent, tgt_sent in zip(sources, targets):
        model_inputs = tokenizer(src_sent, truncation=True, padding=False, max_length=256)
        labels = tokenizer(tgt_sent, truncation=True, padding=False, max_length=256)
        model_inputs["labels"] = labels["input_ids"]
        all_elements.append(model_inputs)
    return all_elements


def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    """Initialize the model and the tokenizer."""
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser(description='This program is about finetuning a frame identification model.')
    parser.add_argument('--train', dest='tr', help='Enter the training data in TSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in TSV format.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    args = parser.parse_args()
    quantization = None
    indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-dist-320M"
    # indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(indic_indic_ckpt_dir, "indic-indic", quantization)
    indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(indic_indic_ckpt_dir, trust_remote_code=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # create the tokenized dataset
    train_dataset = read_lines_from_file(args.tr)
    test_dataset = read_lines_from_file(args.te)
    tokenizer = AutoTokenizer.from_pretrained(indic_indic_ckpt_dir, trust_remote_code=True)
    src_lang, tgt_lang = "hin_Deva", "ory_Orya"
    train_source_sents, train_target_sents = create_source_target_pairs(train_dataset)
    test_source_sents, test_target_sents = create_source_target_pairs(test_dataset)
    train_source_sents = processor.preprocess_batch(train_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    train_target_sents = processor.preprocess_batch(train_target_sents, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True)
    test_source_sents = processor.preprocess_batch(test_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    test_target_sents = processor.preprocess_batch(test_target_sents, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True)
    train_tokenized_dataset = preprocess_function(train_source_sents, train_target_sents, tokenizer)
    test_tokenized_dataset = preprocess_function(test_source_sents, test_target_sents, tokenizer)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.mod,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_strategy='no',
        save_total_limit=1,
        num_train_epochs=args.ep,
        predict_with_generate=True
    )
    data_collator = IndicDataCollator(
        tokenizer=tokenizer,
        model=indic_indic_model,
        padding="longest", # saves padding tokens
        pad_to_multiple_of=8, # better to have it as 8 when using fp16
        label_pad_token_id=-100
    )
    trainer = Seq2SeqTrainer(
        model=indic_indic_model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # if the model is to be trained from the latest checkpoint
    # always put epochs > no_of_epochs when training for the 1st time
    # trainer.train(resume_from_checkpoint=True)
    # to predict and return the class/label with the highest score
    indic_indic_model.save_pretrained(args.mod + '-final')


if __name__ == '__main__':
    main()
