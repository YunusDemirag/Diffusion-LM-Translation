from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import compute_logp
from tokenizers import Tokenizer
from diffusion_translation.parameters import Parameters
import argparse
import wandb
import json
import os
import torch

parser = argparse.ArgumentParser(description="This script trains a diffusion model on a bilingual covost dataset.")

parser.add_argument('-training_args', default='./training_args.json', help='A json file containing the arguments to overwrite.')

args = parser.parse_args()

with open(args.training_args, 'r') as training_args_file:
    training_args = json.load(training_args_file)

COVOST_PATH = "./covost-dataset"
TEXT_DEFAULTS: 'dict[str, str | int | float]' = dict(
    modality='text',
    dataset_name='covost',
    experiment='translation',
    noise_schedule='cosine',
    loss_type='Lsimple',
    dropout=0.1,
    weight_decay=0.0,
    image_size=8,
    hidden_size=128,
    in_channel=16, ## Embedding Dimension
    lr_anneal_steps=400000, ## Training steps
    num_res_blocks=2, ## Not sure
    lr=1e-04, ## Learning rate?
    bsz=64, ## Batch Size
    diff_steps=4000, ## Steps of diffusion
    model_arch='conv-unet',
    emb_scale_factor=1.0, 
    noise_level=0.0, 
    cache_mode='no', 
    use_bert_tokenizer='no',
    padding_mode='block',
    preprocessing_num_workers=1,
    diffusion_models_path = "./improved-diffusion/diffusion_models/"
    #config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
    #model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
    #experiment='gpt2_pre_compress',

)
'''These are the defaults from Diffusion-LMs run_train.py'''

DIFFUSION_DEFAULTS: "dict[str, str | float]" = {
    'seed': 101,
    'data_dir': "",
    'schedule_sampler': "uniform",
    'lr':1e-4,
    'weight_decay':0.0,
    'lr_anneal_steps':0,
    'batch_size':1,
    'microbatch':-1, # -1 disables microbatches
    'ema_rate':"0.9999", # comma-seperated list of EMA values
    'log_interval':50,
    'save_interval':50000,
    'resume_checkpoint':"",
    'use_fp16':False,
    'fp16_scale_growth':1e-3,
    'gradient_clipping':-1.0,
    'eval_interval':2000,
    'checkpoint_path':"diff_models"
}
'''These are the defaults from improved-diffusions train.py'''

MODEL_AND_DIFFUSION_DEFAULTS = model_and_diffusion_defaults()

'''Adjust these for your run.'''

PARAMS = Parameters(**{
    **DIFFUSION_DEFAULTS,
    **MODEL_AND_DIFFUSION_DEFAULTS,
    **TEXT_DEFAULTS,
    **training_args
})
'''This collects all the parameters, as in the Diffusion-Lm repo'''

### From run_train.py
if PARAMS['loss_type'] == 'Lsimple':
    PARAMS.update(use_kl= False, learn_sigma= False)
elif PARAMS['loss_type'] == 'Lhybrid':
    PARAMS.update(use_kl= False, learn_sigma= True)
elif PARAMS['loss_type'] == 'Lvlb':
    PARAMS.update(use_kl= True, learn_sigma= True)
else:
    assert False

def model_path(
    modality,
    padding_mode,
    experiment,
    in_channel,
    model_arch,
    lr,
    weight_decay,
    diff_steps,
    noise_schedule,
    loss_type,
    num_channels,
    num_res_blocks,
    dropout,
    seed,
    notes=None,
    **_
    ):
    MODEL_NAME = f"diff" \
        f"_{modality}" \
        f"_{padding_mode}" \
        f"_{experiment}{in_channel}" \
        f"_{model_arch}" \
        f"_lr{lr}" \
        f"_{weight_decay}" \
        f"_{diff_steps}" \
        f"_{noise_schedule}" \
        f"_{loss_type}" \
        f"_h{num_channels}" \
        f"_s{num_res_blocks}" \
        f"_d{dropout}" \
        f"_sd{seed}" \
        f"{f'_{notes}' if notes else ''}"

    return os.path.join(PARAMS.diffusion_models_path, MODEL_NAME)

MODEL_PATH = model_path(**PARAMS)

PARAMS['checkpoint_path'] = MODEL_PATH

### Environment variables for the training script
os.environ['OPENAI_LOGDIR']=MODEL_PATH
os.environ['TOKENIZERS_PARALLELISM']='false'

set_seed(PARAMS['seed']) 
dist_util.setup_dist() # DEBUG **
logger.configure()

print(PARAMS)

logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **{
        key: PARAMS[key] for key in MODEL_AND_DIFFUSION_DEFAULTS.keys()
    }
)
model.to(dist_util.dev()) #  DEBUG **
# model.cuda() #  DEBUG **

pytorch_total_params = sum(parameter.numel() for parameter in model.parameters())
logger.log(f'the parameter count is {pytorch_total_params}')

schedule_sampler = create_named_schedule_sampler(PARAMS['schedule_sampler'], diffusion)

logger.log(f'saving the hyperparameters to {PARAMS["checkpoint_path"]}/training_args.json')
with open(f'{PARAMS["checkpoint_path"]}/training_args.json', 'w') as hyperparams_file:
    json.dump(PARAMS, hyperparams_file, indent=2)

logger.log("creating data loader...")
print('load data', '*'*50)

tokenizer: 'Tokenizer' = Tokenizer.from_file(f'{COVOST_PATH}/tokenizer_{PARAMS.vocab_size}.json')

embedding_model = torch.nn.Embedding(tokenizer.get_vocab_size(), PARAMS['in_channel'])

torch.save(embedding_model.weight, f'{MODEL_PATH}/embedding_weights_initial.pt')

def create_data_loaders():
    '''Creating a function for this so that unnecessary data can be freed'''
    from improved_diffusion.text_datasets import TextDataset_NoCache
    from torch.utils.data import DataLoader
    import itertools
    from diffusion_translation.datasets import TextDataset_FileBacked

    split_renamings = {
        'train':'data',
        'dev': 'eval_data'
    }

    max_seq_len = PARAMS.image_size ** 2

    from datasets import DatasetDict, Dataset
    from improved_diffusion.text_datasets import _collate_batch_helper

    data = DatasetDict()

    for split in ["test", "train", "dev"]:
        pad_token_id = int(tokenizer.token_to_id('[PAD]'))
        reader = open(f'{COVOST_PATH}/covost_v2.de_en.{split}.txt')

        ## Skip Header
        next(reader)

        encoded = (tokenizer.encode(line).ids for line in reader)

        ## Some lines might be too long. Can't split them up for translation.
        def filtered():
            num_dataset_valid_lines = 0
            num_dataset_filtered_out_lines = 0
            for encoding in encoded:
                if len(encoding) < max_seq_len:
                    num_dataset_valid_lines+=1
                    yield encoding
                else:
                    num_dataset_filtered_out_lines+=1
            num_lines = num_dataset_filtered_out_lines + num_dataset_valid_lines
            print(f'Finished filtering the dataset lines: {num_dataset_filtered_out_lines} out of {num_lines} were too long. ({num_dataset_filtered_out_lines/num_lines*100}\%)')

        encoded_dataset = Dataset.from_dict({
            'input_ids':[ encoding  for encoding in filtered()]
        })

        def pad_function(group_lst):
            group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], pad_token_id, max_seq_len)
            return group_lst

        padded_dataset = encoded_dataset.map(
            pad_function,
            batched=True,
            num_proc=1,
            desc=f'padding',
        )

        data[split]=padded_dataset

        reader.close()

    def data_generator(split, data):
        dataset = TextDataset_NoCache(
            data,
            PARAMS.image_size,
            PARAMS,
            model_arch=PARAMS['model_arch'],
            model_emb=embedding_model.cpu(),
            split=split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=PARAMS['batch_size'],  # 64,
            drop_last=True,
            shuffle=True,
            num_workers=1,
        )

        while True:
            yield from dataloader

    return {
        split_renamings[split]: data_generator(split, data) for split in ['train', 'dev']
    }

data_loaders = create_data_loaders()

embedding_model_cuda = embedding_model.cuda()

def set_mapping_func(args, diffusion):
    '''This mapping func is a parameter in the diffusion instance that will enable it to jointly train the embeddings'''
    print(f'Embedding model: {embedding_model}\n Requires Grad: {embedding_model.weight.requires_grad}')
    mapping_func = partial(compute_logp, args, embedding_model_cuda)
    diffusion.mapping_func = mapping_func

set_mapping_func(PARAMS, diffusion)

def training_params(
    batch_size, 
    microbatch, 
    lr, 
    ema_rate,
    log_interval,
    save_interval,
    resume_checkpoint,
    use_fp16,
    fp16_scale_growth,
    weight_decay,
    lr_anneal_steps,
    checkpoint_path,
    gradient_clipping,
    eval_interval, **_):
    '''Extracts just the training parameters'''
    return dict(
        batch_size=batch_size, 
        microbatch=microbatch, 
        lr=lr, 
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        checkpoint_path=checkpoint_path,
        gradient_clipping=gradient_clipping,
        eval_interval=eval_interval
    )

wandb.init(
    project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
    name=PARAMS['checkpoint_path'],
)
wandb.config.update(PARAMS, allow_val_change=True)

logger.log("training...")
TrainLoop(
    model=model,
    diffusion=diffusion,
    schedule_sampler=schedule_sampler,
    **training_params(**PARAMS),
    **data_loaders
).run_loop()

## Saving the Embedding Model -- Commented out, these ar not touched upon during training
# torch.save(embedding_model.weight, f'{MODEL_PATH}/embedding_weights.pt')
# torch.save(embedding_model_cuda.weight, f'{MODEL_PATH}/embedding_weights_cuda.pt')


# for data_loader in data_loaders:
#     data_loaders[data_loader].close()

wandb.finish(0)