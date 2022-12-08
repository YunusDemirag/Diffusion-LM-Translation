#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
import datasets
import stanza
import spacy_stanza
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise, GPT2VAE, AR_for_cont,\
    Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree, Classifier_Consistency

from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    learned_emb: Optional[str] = field(
        default='no',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/ROCstory',
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='/u/scr/xlisali/diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default='/u/scr/xlisali/e2e_data',
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def get_corpus_wit3(data_args):
    '''Get Wit3 Corpus for the translation Experiment
    in the same format as the RocStories Corpus'''
    from datasets import Dataset, DatasetDict
    from collections import Counter
    from spacy.lang.en import English

    ### Translation Experiment ###
    print('Lodaing from Wit3 Dataset')
    nlp = English()
    tokenizer = nlp.tokenizer
    train_sentence_lst = []
    with open(f'{data_args.train_file}') as train_data_file:
        lines = train_data_file.readlines()
    for line in lines:
        words_in_line = [token.text for token in tokenizer(line)]
        train_sentence_lst.append(words_in_line)
    print(train_sentence_lst[:2])

    valid_sentence_lst = []
    with open(f'{data_args.validation_file}') as valid_data_file:
        lines = valid_data_file.readlines()
    for line in lines:
        words_in_line = [token.text for token in tokenizer(line)]
        valid_sentence_lst.append(words_in_line)
    print(valid_sentence_lst[:2])

    counter = Counter()
    for input_ids in train_sentence_lst:
        counter.update(input_ids)
    for input_ids in valid_sentence_lst:
        counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        vocab_dict[k] = len(vocab_dict)
    print(len(counter), len(vocab_dict))

    
    print(len(vocab_dict), 'derived vocabs')
    train_dataset = Dataset.from_dict({'text': train_sentence_lst})
    valid_dataset = Dataset.from_dict({'text': valid_sentence_lst})
    raw_datasets = DatasetDict({'train': train_dataset, 'validation': valid_dataset})
    raw_datasets.vocab = vocab_dict


    return raw_datasets

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    ###################### LOAD DATASETS & dictionary #########################
    ### Translation Experiment ###
    if model_args.experiment.startswith('roc') and model_args.task == 'translation':
        raw_datasets = get_corpus_wit3(data_args)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    ############# LOAD TOKENIZER ##############
    print('\ninitializing the tokenizer with small vocab\n' + '*'*100)
    print('loading from dataset-specific vocab')
    tokenizer = raw_datasets.vocab
    print(len(tokenizer))
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}

    if model_args.model_name_or_path:

        ############# LOAD MODELS for controllable classifier ##############
        if model_args.experiment in ['roc']:
            if model_args.task == 'classifier':
                import torch
                config.vocab_size = len(tokenizer)
                config.type_vocab_size = 3 
                print('\n Initializing the model from scratch \n' + '*' * 100)
                import json, argparse
                from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
                config_path = os.path.join(model_args.init_emb, "training_args.json")
                print(config_path)
                with open(config_path, 'rb', ) as f:
                    training_args2 = json.load(f)
                # args.__dict__.update(training_args)
                training_args2['sigma_small'] = True
                training_args2['diffusion_steps'] = 200  # 500  # DEBUG
                temp_dict = model_and_diffusion_defaults()
                temp_dict.update(training_args2)
                _, diffusion = create_model_and_diffusion(**temp_dict)
                config.input_emb_dim = model_args.n_embd
                config.train_diff_steps = training_args2['diffusion_steps']
                model = Classifier_Consistency(config=config, diffusion=diffusion,)
                path_save = '/u/scr/nlp/xlisali/predictability/diffusion_models_v7/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long/model750000.pt'
                embedding_weight = torch.load(path_save)['word_embedding.weight']
                print(embedding_weight.shape)
                model.bert.embeddings.word_embeddings.weight = embedding_weight
                model.bert.embeddings.word_embeddings.weight.requires_grad = False
            else:
                config.vocab_size = len(tokenizer)
                print('\n Initializing the model from scratch \n' + '*' * 100)
                model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if (model_args.experiment.startswith('roc')) and model_args.task not in ['data_teacher', 'finetune']:
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                if model_args.task == 'classifier':

                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq1 + seq2 + seqmid] + [1] for (seq1, seq2, seqmid) in
                                 zip(examples['left_text'], examples['right_text'], examples['mid_text'])]

                    type_ids = [[0] + [0] * (len(seq1)+len(seq2)) + [1] * len(seqmid) + [1] for
                                 (seq1, seq2, seqmid) in
                                 zip(examples['left_text'], examples['right_text'], examples['mid_text'])]

                    labels = examples['label']
                    result_dict = {'input_ids': input_ids, 'type_ids':type_ids, 'labels':labels}
                else:
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    result_dict = {'input_ids': input_ids}
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if model_args.padding_mode == 'block':
            block_size = 64
            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                result = {
                    k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
        else:
            def pad_function(group_lst):
                vocab_dict = raw_datasets.vocab

                max_length = 64
                seqlen = 64
                if model_args.task == 'classifier':
                    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'],
                                                                   max_length)
                    group_lst['type_ids'] = _collate_batch_helper(group_lst['type_ids'], 2,
                                                                   max_length)
                    group_lst["labels"] = group_lst["labels"]
                else:
                    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                    group_lst["labels"] = group_lst["input_ids"].copy()

                return group_lst

            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    pad_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"padding",
                )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            print(logits[0].shape, logits[1].shape)
            if type(logits) == tuple:
                return logits[0].argmax(dim=-1)
            else:
                return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    trainer_tokenizer = None if ((model_args.experiment in ['pos', 'synth', 'roc', 'simple-wiki', 'e2e-tgt',
                                                            'e2e-tgt-pos','e2e-tgt-tree', 'e2e-back', 'e2e-back_t2']
                                 or model_args.experiment in ['synth_emb', 'pos_emb', 'roc_emb', 'simple-wiki_emb', 'e2e-tgt_emb'])
                                 and model_args.task not in ['data_teacher', 'finetune']) \
                        else tokenizer
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=trainer_tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
