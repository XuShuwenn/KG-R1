# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Utils for tokenization."""

import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None, except for LLaMA models which use unk_token.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    # Use UNK token for LLaMA models, EOS for others
    is_llama = hasattr(tokenizer, 'name_or_path') and 'llama' in str(tokenizer.name_or_path).lower()
    
    if tokenizer.pad_token_id is None:
        if is_llama:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            warnings.warn(f"LLaMA tokenizer.pad_token_id is None. Now set to {tokenizer.unk_token_id} (UNK)", stacklevel=1)
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    
    if tokenizer.pad_token is None:
        if is_llama:
            tokenizer.pad_token = tokenizer.unk_token
            warnings.warn(f"LLaMA tokenizer.pad_token is None. Now set to {tokenizer.unk_token} (UNK)", stacklevel=1)
        else:
            tokenizer.pad_token = tokenizer.eos_token
            warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    import os
    import json
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    
    # Load chat template with priority: tokenizer_config.json > chat_template.jinja
    # Only load if tokenizer doesn't already have a chat template, or if we need to override
    if isinstance(name_or_path, str) and os.path.exists(name_or_path):
        chat_template = None
        chat_template_loaded = False
        
        # Priority 1: Try to load from tokenizer_config.json
        tokenizer_config_path = os.path.join(name_or_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            try:
                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    tokenizer_config = json.load(f)
                    chat_template = tokenizer_config.get("chat_template", None)
                    if chat_template:
                        # Apply chat template from tokenizer_config
                        tokenizer.chat_template = chat_template
                        chat_template_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load chat template from tokenizer_config.json: {e}", stacklevel=1)
        
        # Priority 2: Fallback to chat_template.jinja if not found in tokenizer_config
        if not chat_template_loaded:
            chat_template_jinja_path = os.path.join(name_or_path, "chat_template.jinja")
            if os.path.exists(chat_template_jinja_path):
                try:
                    with open(chat_template_jinja_path, "r", encoding="utf-8") as f:
                        chat_template = f.read()
                        # Apply chat template from jinja file
                        tokenizer.chat_template = chat_template
                        chat_template_loaded = True
                except Exception as e:
                    warnings.warn(f"Failed to load chat template from chat_template.jinja: {e}", stacklevel=1)
    
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
