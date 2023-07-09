from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
#from transformers.utils import to_py_obj
import re
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# I create another class VLT5Tokenizer extending it to add <vis_extra_id_{}>
# def to_py_obj(obj):
#     """
#     Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
#     """
#     if isinstance(obj, (dict, UserDict)):
#         return {k: to_py_obj(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [to_py_obj(o) for o in obj]
#     elif is_tf_tensor(obj):
#         return obj.numpy().tolist()
#     elif is_torch_tensor(obj):
#         return obj.detach().cpu().tolist()
#     elif is_jax_tensor(obj):
#         return np.asarray(obj).tolist()
#     elif isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
#         return obj.tolist()
#     else:
#         return obj


class VLT5Tokenizer(T5Tokenizer):

    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        ocr_extra_ids= 0, #Added by Mingyang
        num_extra_ids= 0, #Added by Mingyang
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])
        
        if ocr_extra_ids > 0:
            additional_special_tokens.extend(["<ocr_extra_id_{}>".format(i) for i in range(ocr_extra_ids)])
        
        if num_extra_ids > 0:
            additional_special_tokens.extend(["<num_extra_id_{}>".format(i) for i in range(num_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids
        self._ocr_extra_ids = ocr_extra_ids
        self._num_extra_ids = num_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        #return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids
        return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids + self._ocr_extra_ids + self._num_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            #return self.vocab_size - num - 1 - self._vis_extra_ids
            return self.vocab_size - num - 1 - self._vis_extra_ids - self._ocr_extra_ids -self._num_extra_ids
        elif token.startswith("<vis_extra_id_"):
            match = re.match(r"<vis_extra_id_(\d+)>", token)
            num = int(match.group(1))
            #return self.vocab_size - num - 1
            return self.vocab_size - num - 1 - self._ocr_extra_ids - self._num_extra_ids
        elif token.startswith("<ocr_extra_id_"):
            match = re.match(r"<ocr_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._num_extra_ids
        elif token.startswith("<num_extra_id_"):
            match = re.match(r"<num_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1

        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index >  self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids +self._ocr_extra_ids - 1:
                token = "<num_extra_id_{}>".format(self.vocab_size - 1 - index)
            elif index >  self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids-1:
                token = "<ocr_extra_id_{}>".format(self.vocab_size - self.num_extra_ids - 1 - index)
            if index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<vis_extra_id_{}>".format(self.vocab_size - self.num_extra_ids - self.ocr_extra_ids - 1 - index)
                #token = "<vis_extra_id_{}>".format(self.vocab_size - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self.num_extra_ids - self.ocr_extra_ids - self._vis_extra_ids - 1 - index)
                #token = "<extra_id_{}>".format(self.vocab_size - self._vis_extra_ids - 1 - index)
        return token
    
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False, num_token_values=None):
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for i, index in enumerate(ids):
            index = int(index)
            if skip_special_tokens and index != 32300 and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            elif index == 32300:
                assert num_token_values is not None
                tokens.append("%.3f"%num_token_values[i])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, num_token_values=None, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`str`: The decoded sentence.
        """
        # Convert inputs to python lists
        #token_ids = to_py_obj(token_ids)
        if type(token_ids) is not list:
            token_ids = token_ids.detach().cpu().tolist()

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            num_token_values=num_token_values,
            **kwargs,
        )
    def _decode(self,token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, spaces_between_special_tokens=True, num_token_values=None):
        
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens, num_token_values=num_token_values)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

# Below are for Rust-based Fast Tokenizer

from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from typing import Any, Dict, List, Optional, Tuple, Union


class VLT5Converter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]

        num_vis_extra_ids = self.original_tokenizer._vis_extra_ids
        vocab += [("<vis_extra_id_{}>".format(i), 0.0)
                  for i in range(num_vis_extra_ids - 1, -1, -1)]
        
        num_ocr_extra_ids = self.original_tokenizer._ocr_extra_ids
        vocab += [("<ocr_extra_id_{}>".format(i), 0.0)
                  for i in range(num_ocr_extra_ids - 1, -1, -1)]

        num_num_extra_ids = self.original_tokenizer._num_extra_ids
        vocab += [("<num_extra_id_{}>".format(i), 0.0)
                  for i in range(num_num_extra_ids - 1, -1, -1)]

        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_vlt5tokenizer(vlt5tokenizer):
    return VLT5Converter(vlt5tokenizer).converted()


class VLT5TokenizerFast(T5TokenizerFast):

    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]
    slow_tokenizer_class = VLT5Tokenizer

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        ocr_extra_ids=0, #changed by mingyang
        num_extra_ids=0, #Default should be 1
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])
        
        if ocr_extra_ids > 0:
            additional_special_tokens.extend(["<ocr_extra_id_{}>".format(i) for i in range(ocr_extra_ids)])
        
        if num_extra_ids > 0:
            additional_special_tokens.extend(["<num_extra_id_{}>".format(i) for i in range(num_extra_ids)])

        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            ocr_extra_ids=ocr_extra_ids,
            num_extra_ids=num_extra_ids,
            # additional_special_tokens=additional_special_tokens,
            **kwargs
        )
        fast_tokenizer = convert_slow_vlt5tokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer
        self._slow_tokenizer = slow_tokenizer
        # print(slow_tokenizer.vocab_size)
        # print(self._tokeni)
        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            ocr_extra_ids=ocr_extra_ids,
            num_extra_ids=num_extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids
        self._ocr_extra_ids = ocr_extra_ids
        self._num_extra_ids = num_extra_ids
    def batch_decode(self,sequences,skip_special_tokens=False,clean_up_tokenization_spaces=True, num_token_values=None, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (:obj:`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`List[str]`: The list of decoded sentences.
        """
        if num_token_values is not None:
            return [
                self.decode(
                    seq,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    num_token_values = num_seq,
                    **kwargs,
                )
                for seq, num_seq in zip(sequences, num_token_values)
            ]
        else:
            return[
                self.decode(
                    seq,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    **kwargs,
                )
                for seq in sequences
            ]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, num_token_values=None, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`str`: The decoded sentence.
        """
        # Convert inputs to python lists
        #token_ids = to_py_obj(token_ids)
        #assume this a torch tensor
        token_ids = token_ids.detach().cpu().tolist()

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            num_token_values=num_token_values,
            **kwargs,
        )

    def _decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, num_token_values=None, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if num_token_values is not None:
            text = self._slow_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, num_token_values=num_token_values)
        else:
            text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text




    # @property
    # def vocab_size(self):
    #     return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids + self._ocr_extra_ids
