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
model = model.to(cuda)

embedding_model : nn.Embedding = model.word_embedding

################## SAMPLING ##################
##############################################

from improved_diffusion.test_util import denoised_fn_round
from functools import partial

model_kwargs = {}

sample_shape = (PARAMS.batch_size, PARAMS.seqlen, PARAMS.in_channel, )

sample: torch.Tensor = diffusion.p_sample_loop(
    model,
    sample_shape,
    denoised_fn=partial(denoised_fn_round, PARAMS, embedding_model) if "clamp" in PARAMS.keys() and PARAMS.clamp == "clamp" else None,
    clip_denoised=PARAMS['clip_denoised'],
    model_kwargs=model_kwargs,
    top_p = PARAMS.top_p,
    device=cuda,
    progress=True
)

print(sample[0])

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
print(indices[0])
for seq in indices:
    numpy_sequence = seq.cpu().numpy()
    tokens = tokenizer.decode(numpy_sequence.squeeze(-1))
    decoded_outputs.append(tokens)

decoded_outputs_in_lines = (decoded_output + '\n' for decoded_output in decoded_outputs)

with open(f'{output_directory}/samples_e2e_decoded.txt', 'w') as output_file:
    output_file.writelines(decoded_outputs_in_lines)