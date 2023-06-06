from tokenizers import Tokenizer
from diffusion_translation.parameters import Parameters
from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from improved_diffusion import logger
import torch.nn as nn
import torch
import json
import argparse

parser = argparse.ArgumentParser(
    prog="Sample and Decode",
    description="Used to sample from a diffusion model, and decode using k-nearest-neighbours"
)

parser.add_argument(
    "-model_path", 
    help="The relative path to the directory the model is saved in."
)

parser.add_argument(
    "-sampling_args", 
    default="./sampling_args.json",
    required=False,
    help="Path to a json file with the arguments for sampling."
)

args = parser.parse_args()

# load configurations.
with open(f'{args.model_path}/training_args.json', 'rb', ) as training_args_file:
    training_args = json.load(training_args_file)

with open(args.sampling_args, 'r') as sampling_args_file:
    sampling_args = json.load(sampling_args_file)

PARAMS = Parameters(**{
    **training_args,
    **sampling_args,
    **{"model_path": args.model_path}
})

output_directory = PARAMS.output_directory if 'output_directory' in PARAMS.keys() else PARAMS.model_path

################ LOADING MODELS AND SETUP ################
##########################################################

logger.configure()

cuda = torch.full([1],1).cuda().device

tokenizer = Tokenizer.from_file(PARAMS.tokenizer)

padding_token = tokenizer.token_to_id('[PAD]')

model, diffusion = create_model_and_diffusion(
    **{key: PARAMS[key] for key in model_and_diffusion_defaults().keys()}
)

model.load_state_dict(torch.load(f'{PARAMS.model_path}/{PARAMS.model_file}'))
model = model.cuda()

embedding_model : nn.Embedding = model.word_embedding

########### CREATING DATA LOADERS ############
##############################################

COVOST_PATH = "./covost-dataset"

def create_data_loader():
    '''Creating a function for this so that unnecessary data can be freed'''
    from improved_diffusion.text_datasets import TextDataset_NoCache
    from torch.utils.data import DataLoader

    # As in Diffusion-LM
    max_seq_len = PARAMS.image_size ** 2

    from datasets import DatasetDict, Dataset
    from improved_diffusion.text_datasets import _collate_batch_helper

    data = DatasetDict()

    for split in ["test"]:
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

    # The dataloader handles data on the cpu, so we need to clone the embedding model
    embedding_model_cpu = nn.Embedding(embedding_model.num_embeddings, embedding_model.embedding_dim)
    embedding_model_cpu.load_state_dict(embedding_model.state_dict())
    embedding_model_cpu.cpu()

    def data_generator(split, data):
        dataset = TextDataset_NoCache(
            data,
            PARAMS.image_size,
            PARAMS,
            model_arch=PARAMS['model_arch'],
            model_emb=embedding_model_cpu,
            split=split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=PARAMS['batch_size'],  # 64,
            drop_last=False,
            shuffle=False,
            num_workers=1,
        )

        for batch in dataloader:
            yield batch

    return data_generator(split, data)

data_loader = create_data_loader()

################## SAMPLING ##################
##############################################

from improved_diffusion.test_util import denoised_fn_round
from functools import partial

model_kwargs = {}

sample_shape = (PARAMS.batch_size, PARAMS.seqlen, PARAMS.in_channel, )

output_file = open(f'{output_directory}/samples_translation_decoded.txt', 'w')
raw_output_file = open(f'{output_directory}/samples_raw_decoded.txt', 'w')

json_encoder = json.JSONEncoder()

for batch in data_loader:

    ## Adjust sample shape to batch size (last batch might be smaller)
    if batch[0].shape[0] != PARAMS.batch_size:
        sample_shape_adjusted = (batch[0].shape[0], PARAMS.seqlen, PARAMS.in_channel, )
    else:
        sample_shape_adjusted = sample_shape

    seperator_id: int = tokenizer.token_to_id('[SEP]')
    batch_input_ids = torch.tensor(batch[1]["input_ids"], device=cuda)
    encoded_reference_sequence = embedding_model.cuda()(batch_input_ids)
    seperator_matrix_indices = (batch_input_ids == seperator_id).nonzero()
    seperator_indices = seperator_matrix_indices[:,1:2]
    seperator_indices_broadcasted = seperator_indices.expand(batch_input_ids.shape)
    position_indices = torch.tensor(range(PARAMS.seqlen), device=cuda).expand(batch_input_ids.shape)
    mask = (seperator_indices_broadcasted < position_indices)

    sources_ids = batch_input_ids.clone().detach()
    sources_ids[mask] = padding_token

    references_ids = batch_input_ids.clone().detach()
    references_ids[~mask] = padding_token

    sample_generator = diffusion.p_sample_loop_progressive_infill(
        model,
        sample_shape_adjusted,
        encoded_reference_sequence,
        mask,
        denoised_fn=partial(denoised_fn_round, PARAMS, embedding_model.cuda()) if "clamp" in PARAMS.keys() and PARAMS.clamp == "clamp" else None,
        clip_denoised=PARAMS['clip_denoised'],
        model_kwargs=model_kwargs,
        device=cuda,
        progress=True,
        greedy=False
    )

    for final in sample_generator:
        sample = final['sample']

    # The sampling apparently moves the model to the cpu
    model = model.cuda()

    ################## DECODING ##################
    ##############################################

    import numpy as np

    samples: np.ndarray = np.concatenate([sample.cpu().numpy()], axis=0)

    decoded_outputs = []

    print(f'Decoding for e2e, Sample shape: {sample.shape}, Sample dtype: {sample.dtype}')

    x_ts = torch.tensor(sample, device=cuda, dtype=torch.float32)
    '''As in paper, x_t is the final output of the diffusion'''

    if PARAMS.model_arch == 'conv-unet':
        x_ts = x_ts.view(x_ts.size(0), -1, x_ts.size(-1))

    logits = model.get_logits(x_ts)  # bsz, seqlen, vocab_size | -|Hidden_Repr-Embedding|
    most_probable_tokens = torch.topk(logits, k=1, dim=-1) 
    indices = most_probable_tokens.indices # bsz, seqlen, 1

    decoded_outputs_raw = []
    print(indices[0])
    for seq in indices:
        numpy_sequence = seq.cpu().numpy()
        tokens = tokenizer.decode(numpy_sequence.squeeze(-1))
        decoded_outputs_raw.append(tokens)

    raw_output_file.writelines((decoded_output_raw + '\n' for decoded_output_raw in decoded_outputs_raw))
    raw_output_file.flush()

    indices[~mask] = padding_token # Replacing the source with padding so the translation is left
    for seq in indices:
        numpy_sequence = seq.cpu().numpy()
        tokens = tokenizer.decode(numpy_sequence.squeeze(-1))
        decoded_outputs.append(tokens)

    references = (tokenizer.decode(reference_ids.cpu().numpy()) for reference_ids in references_ids)
    sources = (tokenizer.decode(source_ids.cpu().numpy()) for source_ids in sources_ids)
    recover_reference_source_texts = zip(decoded_outputs, references, sources)

    output_objects = (
        {
            "recover": decoded_output,
            "reference": reference,
            "source":source
        } for decoded_output, reference, source in recover_reference_source_texts
    )

    output_strings = (f'{json_encoder.encode(output_object)}\n' for output_object in output_objects)

    output_file.writelines(output_strings)
    output_file.flush()

output_file.close()