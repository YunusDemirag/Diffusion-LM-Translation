import numpy as np
import itertools
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset
import torch
from torch.nn import Embedding

class TextDataset_FileBacked(IterableDataset):
    '''This is a dataset implementation made to work seamlessly with the text datasets for Diffusion-LM using streams instead of holding the data in memory.'''
    def __init__(self, image_size, tokenizer: Tokenizer, embedding_model: Embedding, file=None, model_arch='conv-unet', eigen_transform=None,
                 mapping_func=None, model_emb=None, **kwargs):
        super().__init__()
        self.resolution = image_size
        self.model_arch = model_arch
        self.kwargs = kwargs
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb: Embedding = embedding_model
        self.tokenizer = tokenizer
        self.max_seq_len = image_size ** 2
        self.reader = None
        self.initialized = False
        self.generator = None
        self.file = file

        if self.model_arch == 'conv-unet':
            self.model_arch_process_func = self._model_arch_conv_unet_process
        elif self.model_arch == '1d-unet':
            self.model_arch_process_func = self._model_arch_1d_unet_process
        else:
            self.model_arch_process_func = self._model_arch_default_process

        
        
        if file:
            self.open(file)

    def open(self, file: str):
        '''Open an underlying file.'''

        self.file = file

        if self.initialized:
            raise IOError("Can't open this dataset twice")

        pad_token_id = int(self.tokenizer.token_to_id('[PAD]'))
        self.reader = open(file, 'r', buffering=4194304)

        ## Skip header
        next(self.reader)

        encoded = (self.tokenizer.encode(line).ids for line in self.reader)


        ## Some lines might be too long. Can't split them up for translation.
        def filtered():
            num_dataset_valid_lines = 0
            num_dataset_filtered_out_lines = 0
            for encoding in encoded:
                if len(encoding) < self.max_seq_len:
                    num_dataset_valid_lines+=1
                    yield encoding
                else:
                    num_dataset_filtered_out_lines+=1
            num_lines = num_dataset_filtered_out_lines + num_dataset_valid_lines
            print(f'Finished filtering the dataset lines: {num_dataset_filtered_out_lines} out of {num_lines} were too long. ({num_dataset_filtered_out_lines/num_lines*100}\%)')

        filtered_encodings_as_tensors = (torch.tensor(encoding, dtype=torch.int64) for encoding in filtered())
        sequence_generator = itertools.repeat(torch.full([self.max_seq_len], pad_token_id, dtype=torch.int64))
        def padded_encodings():
            for encoding in filtered_encodings_as_tensors:
                sequence = next(sequence_generator)
                sequence[:len(encoding)] = encoding
                yield sequence

        self.generator = padded_encodings()
        self.initialized = True

    def __iter__(self):

        if not self.initialized:
            raise "The Dataset has not been initialized"

        while True:

            sequence = next(self.generator, None)
            if sequence == None:
                self.initialized = False
                self.generator.close()
                self.reader.close()
                self.open(self.file)
                sequence = next(self.generator)


            with torch.no_grad():
                hidden_state = self.model_emb(sequence)
                result = self.model_arch_process_func(hidden_state=hidden_state, sequence=sequence)
            
            yield result
        

    def _model_arch_conv_unet_process(self, hidden_state, sequence):
        arr = np.array(hidden_state, dtype=np.float32).reshape(self.resolution, self.resolution, -1)
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)
        if hasattr(self.kwargs, 'noise_level') and self.kwargs.noise_level > 0:
            arr = arr + self.kwargs.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

        out_dict = {}
        out_dict['input_ids'] = np.array(sequence)
        return np.transpose(arr, [2, 0, 1]), out_dict

    def _model_arch_1d_unet_process(self, hidden_state, sequence):
        arr = np.array(hidden_state, dtype=np.float32)  # seqlen, dim
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)
        if hasattr(self.kwargs, 'noise_level') and self.kwargs.noise_level > 0:
            arr = arr + self.kwargs.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
        arr = np.transpose(arr, [1, 0])
        out_dict = {}
        out_dict['input_ids'] = np.array(sequence)
        return arr, out_dict

    def _model_arch_default_process(self, hidden_state, sequence):
        arr = np.array(hidden_state, dtype=np.float32)
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)

        if hasattr(self.kwargs, 'noise_level') and self.kwargs.noise_level > 0:
            arr = arr + self.kwargs.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
        out_dict = {}
        out_dict['input_ids'] = np.array(sequence)
        # if self.kwargs.experiment_mode == 'conditional_gen':
        #     out_dict['src_ids'] = np.array(self.text_datasets[self.split][idx]['src_ids'])
        #     out_dict['src_mask'] = np.array(self.text_datasets[self.split][idx]['src_mask'])
        return arr, out_dict

    def close(self):
        '''Closes the underlying stream'''
        self.reader.close()