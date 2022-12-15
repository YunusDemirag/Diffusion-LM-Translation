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

parser.add_argument('-resume_model', help='The model directory')
parser.add_argument('-resume_checkpoint', help='The checkpoint you want to continue from')
parser.add_argument('-run_id', help="WAND Run ID", required=False)

args = parser.parse_args()

with open(f'{args.resume_model}/training_args.json', 'r') as training_args_file:
    training_args = json.load(training_args_file)

COVOST_PATH = "./covost-dataset"

PARAMS = Parameters(**{
    **training_args
})
'''This collects all the parameters, as in the Diffusion-Lm repo'''

MODEL_PATH = args.resume_model

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
        key: PARAMS[key] for key in model_and_diffusion_defaults().keys()
    }
)
model.to(dist_util.dev()) #  DEBUG **
# model.cuda() #  DEBUG **

pytorch_total_params = sum(parameter.numel() for parameter in model.parameters())
logger.log(f'the parameter count is {pytorch_total_params}')

schedule_sampler = create_named_schedule_sampler(PARAMS['schedule_sampler'], diffusion)

logger.log("creating data loader...")
print('load data', '*'*50)

tokenizer: 'Tokenizer' = Tokenizer.from_file(f'{COVOST_PATH}/tokenizer_{PARAMS.vocab_size}.json')

embedding_model = torch.nn.Embedding(tokenizer.get_vocab_size(), PARAMS['in_channel'])

embedding_model.weight = torch.load(f'{MODEL_PATH}/embedding_weights_initial.pt')

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

def training_params(
    batch_size, 
    microbatch, 
    lr, 
    ema_rate,
    log_interval,
    save_interval,
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
    resume=True,
    id=args.run_id
)

logger.log("resuming training...")
TrainLoop(
    model=model,
    diffusion=diffusion,
    schedule_sampler=schedule_sampler,
    resume_checkpoint=f'{MODEL_PATH}/model{args.resume_checkpoint}.pt',
    **training_params(**PARAMS),
    **data_loaders
).run_loop()

## Saving the Embedding Model -- Commented out, these ar not touched upon during training
# torch.save(embedding_model.weight, f'{MODEL_PATH}/embedding_weights.pt')
# torch.save(embedding_model_cuda.weight, f'{MODEL_PATH}/embedding_weights_cuda.pt')


# for data_loader in data_loaders:
#     data_loaders[data_loader].close()

wandb.finish(0)