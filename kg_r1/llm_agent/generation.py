import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json # Added for parsing JSON queries and formatting fallback
from functools import lru_cache
from kg_r1.search.error_types import KGErrorType
from kg_r1.kgqa_bridge import KGQASparqlAdapter

# Import prompt utilities for continuation and force-answer prompts
try:
    from verl.trainer.ppo.prompts import build_continuation_prompt, FORCE_ANSWER_PROMPT
except ImportError:
    # Fallback to kgqa_agent prompts if verl is not available
    try:
        from kgqa_agent.prompts.prompts import build_continuation_prompt, FORCE_ANSWER_PROMPT
    except ImportError:
        # Define fallback functions if both imports fail
        def build_continuation_prompt(query_results: str) -> str:
            return f'<information>Here are the query results:\n{query_results}\n</information>\n\nReview the results. If the answer is found, provide it in <answer> tags. Otherwise, continue reasoning in <think> tags and issue your next <kg-query>.\n\nReminder: After `get_relations`, you must rank all returned relations and call `get_triples` with the full ranked list. We will execute a query for the top 4.'
        FORCE_ANSWER_PROMPT = """You have reached the maximum number of queries. Based on the information gathered, provide your final answer in <answer> tags.
Strict Format: <answer>["Answer1", "Answer2"]</answer>. The answer(s) must be concise entity names copied exactly from the KG results.
"""

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3 # Note: topk is not directly used by the KG server's current actions
    # NEW: Dataset-specific server URLs (optional, fallback to environment variables or defaults)
    simpleqa_server_url: str = None  # Default: FB2M server on port 9001
    cwq_server_url: str = None 
    webqsp_server_url: str = None
    metaqa_server_url: str = None  # Default: MetaQA server on port 9002
    use_sparql_bridge: bool = False
    sparql_endpoint: str = None
    kgqa_relation_filter_model: str = None
    kgqa_max_calls: int = None
    kgqa_top_k: int = None
    kgqa_timeout: int = 15  # Timeout for SPARQL queries (seconds)
    kgqa_thread_pool_size: int = 40  # Max workers for per-turn SPARQL parallelism
    kgqa_thread_timeout: Optional[float] = None  # Optional timeout per query future

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        
        # Handle tokenizer-specific configurations
        self._setup_tokenizer_compatibility()
        
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.use_sparql_bridge = bool(config.use_sparql_bridge)
        self.kgqa_thread_pool_size = max(
            1, int(getattr(config, "kgqa_thread_pool_size", 40) or 40)
        )
        self.kgqa_thread_timeout = getattr(config, "kgqa_thread_timeout", None)
        # Store kgqa_timeout for fallback when thread_timeout is None
        self.kgqa_timeout = getattr(config, "kgqa_timeout", 15) or 15
        self.kgqa_adapter: KGQASparqlAdapter | None = None
        if self.use_sparql_bridge:
            adapter_endpoint = config.sparql_endpoint or config.search_url
            if not adapter_endpoint:
                raise ValueError("SPARQL bridge enabled but no sparql_endpoint/search_url provided.")
            adapter_top_k = config.kgqa_top_k or config.topk
            adapter_max_calls = config.kgqa_max_calls or 10  # Default to 10 if not specified
            adapter_filter_model = config.kgqa_relation_filter_model
            adapter_timeout = config.kgqa_timeout or 15  # Default to 15s if not specified
            self.kgqa_adapter = KGQASparqlAdapter(
                sparql_endpoint=adapter_endpoint,
                timeout=adapter_timeout,
                kg_top_k=adapter_top_k,
                max_calls=adapter_max_calls,
                relation_filter_model=adapter_filter_model,
            )
            filter_msg = (
                adapter_filter_model if adapter_filter_model else "disabled"
            )
            print(
                f"[KG_BRIDGE] Initialized kgqa_agent SPARQL adapter @ {adapter_endpoint} "
                f"(max_calls={adapter_max_calls}, timeout={adapter_timeout}s, relation_filter_model={filter_msg})"
            )
        
        # Track if we've already logged routing for each dataset
        self._routing_logged = set()

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Pre-compile regex patterns for optimization
        self._init_regex_patterns()
        
        # Initialize server routing configuration (only for FastAPI mode, not for SPARQL bridge)
        if not self.use_sparql_bridge:
            self._init_server_routing()

    def _setup_tokenizer_compatibility(self):
        """
        Setup tokenizer compatibility for different model families.
        Handles Llama2, Llama3, and Qwen2.5 tokenizers.
        """
        # Get tokenizer class name and model name if available
        tokenizer_class = self.tokenizer.__class__.__name__
        model_name_or_path = getattr(self.tokenizer, 'name_or_path', '')
        
        # Detect tokenizer type based on class name (primary) and model path (secondary)
        # Priority: tokenizer_class > model_name_or_path
        # Note: model_name_or_path may contain "LLaMA-Factory" in the path, so we check tokenizer_class first
        is_llama = False
        is_qwen = False
        
        # Check tokenizer class first (most reliable indicator)
        tokenizer_class_lower = tokenizer_class.lower()
        model_path_lower = model_name_or_path.lower()
        
        # Check for Llama tokenizers (Llama2 or Llama3)
        # Only check model path if tokenizer class doesn't clearly indicate the type
        if 'llama' in tokenizer_class_lower:
            is_llama = True
            print(f"[TOKENIZER] Detected Llama tokenizer: {tokenizer_class} (model: {model_name_or_path})")
        # Check for Qwen tokenizers
        elif 'qwen' in tokenizer_class_lower:
            is_qwen = True
            print(f"[TOKENIZER] Detected Qwen tokenizer: {tokenizer_class} (model: {model_name_or_path})")
        # Fallback: check model path if tokenizer class doesn't match
        # But be careful: model path may contain "LLaMA-Factory" which is not the model type
        elif 'llama' in model_path_lower and 'qwen' not in model_path_lower:
            # Only treat as Llama if path contains "llama" but NOT "qwen"
            # Extract just the model name part (last component of path) to avoid false positives
            model_name = model_path_lower.split('/')[-1] if '/' in model_path_lower else model_path_lower
            if 'llama' in model_name:
                is_llama = True
                print(f"[TOKENIZER] Detected Llama tokenizer (from model path): {tokenizer_class} (model: {model_name_or_path})")
            elif 'qwen' in model_name:
                is_qwen = True
                print(f"[TOKENIZER] Detected Qwen tokenizer (from model path): {tokenizer_class} (model: {model_name_or_path})")
        elif 'qwen' in model_path_lower:
            is_qwen = True
            print(f"[TOKENIZER] Detected Qwen tokenizer (from model path): {tokenizer_class} (model: {model_name_or_path})")
        else:
            # Raise error for unsupported tokenizers
            raise ValueError(
                f"Unsupported tokenizer: {tokenizer_class} (model: {model_name_or_path}). "
                f"Only Llama2, Llama3, and Qwen2.5 tokenizers are supported."
            )
        
        # Handle Llama-specific setup
        if is_llama:
            # Llama tokenizers don't have a pad_token by default
            if self.tokenizer.pad_token is None:
                print("[TOKENIZER] Llama tokenizer has no pad_token, setting pad_token = eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                print(f"[TOKENIZER] Llama tokenizer already has pad_token: {repr(self.tokenizer.pad_token)}")
        
        # Qwen tokenizers already have proper pad_token setup, no changes needed
        elif is_qwen:
            print(f"[TOKENIZER] Qwen tokenizer pad_token: {repr(self.tokenizer.pad_token)} (id: {self.tokenizer.pad_token_id})")
            # No modifications needed for Qwen - preserve existing behavior
        
        # Verify pad_token is now set
        if self.tokenizer.pad_token_id is None:
            raise ValueError(f"Failed to set pad_token_id for {tokenizer_class}")

    def _init_server_routing(self):
        """Initialize server URL routing configuration for different datasets."""
        # Priority: Config > Environment Variables > Defaults
        self.server_routing = {
            "simpleqa": os.environ.get("SIMPLEQA_SERVER_URL", "http://127.0.0.1:9001/retrieve"),
            "cwq": os.environ.get("CWQ_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "webqsp": os.environ.get("WEBQSP_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "multitq": os.environ.get("MULTITQ_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "metaqa": os.environ.get("METAQA_SERVER_URL", "http://127.0.0.1:9002/retrieve"),
            "grailqa": os.environ.get("GRAILQA_SERVER_URL", "http://127.0.0.1:9000/retrieve"),
            "trex": os.environ.get("TREX_SERVER_URL", "http://127.0.0.1:9011/retrieve"),
            "qald10en": os.environ.get("QALD10EN_SERVER_URL", "http://127.0.0.1:9010/retrieve"),
            "zero_shot_re": os.environ.get("ZERO_SHOT_RE_SERVER_URL", "http://127.0.0.1:9012/retrieve"),
        }
        
        # Fallback server URL (original behavior)
        self.fallback_server_url = self.config.search_url or "http://127.0.0.1:8001/retrieve"
        
        print(f"[KG_ROUTING] Server routing configuration:")
        for dataset, url in self.server_routing.items():
            print(f"  {dataset} -> {url}")
        print(f"  fallback -> {self.fallback_server_url}")

    def _get_server_url_for_dataset(self, dataset_name: str) -> str:
        """
        Get the appropriate server URL for a given dataset.
        
        Args:
            dataset_name: Name of the dataset (simpleqa, cwq, webqsp, etc.)
            
        Returns:
            Server URL for the dataset
        """
        # This method should only be called in FastAPI mode (not SPARQL bridge mode)
        if not hasattr(self, 'server_routing') or not hasattr(self, 'fallback_server_url'):
            raise RuntimeError(
                "Server routing not initialized. This method should only be called in FastAPI mode, "
                "not in SPARQL bridge mode."
            )
        
        if not dataset_name:
            return self.fallback_server_url
            
        # Normalize dataset name to lowercase
        dataset_key = dataset_name.lower().strip()
        
        # Return specific server URL or fallback
        server_url = self.server_routing.get(dataset_key, self.fallback_server_url)
        
        # Log routing decision for debugging (only once per dataset)
        if dataset_key not in self._routing_logged:
            self._routing_logged.add(dataset_key)
            if dataset_key in self.server_routing:
                print(f"[KG_ROUTING] {dataset_name} -> {server_url}")
            else:
                print(f"[KG_ROUTING] Unknown dataset '{dataset_name}', using fallback -> {server_url}")
            
        return server_url

    def _build_bridge_session_keys(self, meta_info: Dict[str, Any] | None, count: int) -> List[str]:
        """Ensure per-example session keys for kgqa adapter."""
        if not self.use_sparql_bridge:
            return []

        if meta_info is None:
            meta_info = {}

        keys = meta_info.get("bridge_session_keys")
        if isinstance(keys, list) and len(keys) == count:
            return keys

        sample_ids = meta_info.get("sample_ids", [])
        session_keys: List[str] = []
        for idx in range(count):
            base_id = sample_ids[idx] if idx < len(sample_ids) else f"sparql_sample_{idx:06d}"
            session_keys.append(f"{base_id}__idx_{idx}")
        meta_info["bridge_session_keys"] = session_keys
        return session_keys

    def _reset_bridge_session(self, session_key: str) -> None:
        if self.use_sparql_bridge and self.kgqa_adapter and session_key:
            self.kgqa_adapter.reset(session_key)

    def _init_regex_patterns(self):
        """Pre-compile all regex patterns for performance optimization."""
        # Patterns for postprocess_predictions
        # Note: <search> tag is not a valid tag in kgqa_agent mode, removed
        # Use negative lookahead to prevent cross-tag matching (match content that doesn't contain opening tag)
        self.kg_query_pattern1 = re.compile(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', re.DOTALL | re.IGNORECASE)  
        self.kg_query_pattern2 = re.compile(r'<kg-query\s+([^>]+)\s*/>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>((?:(?!<answer>).)*?)</answer>', re.DOTALL | re.IGNORECASE)
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        
        # Patterns for _parse_kg_query
        self.prefix_patterns = [
            re.compile(r'^kg-query\s+execute\s*["\']?', re.IGNORECASE),
            re.compile(r'^kg-query\s+', re.IGNORECASE), 
            re.compile(r'^function_name\s*\(\s*', re.IGNORECASE),
            re.compile(r'^query\s*[:\s]+', re.IGNORECASE),
        ]
        self.nested_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*(get_[a-zA-Z_]+\([^)]+\))\s*\)$')
        # Strict patterns for single-entity functions (new: get_relations_in/out, legacy: get_head/tail_relations)
        self.get_relations_quoted_pattern = re.compile(r'^(get_relations|get_relations_in|get_relations_out|get_head_relations|get_tail_relations)\s*\(\s*"([^"]+)"\s*\)$')
        self.get_relations_unquoted_pattern = re.compile(r'^(get_relations|get_relations_in|get_relations_out|get_head_relations|get_tail_relations)\s*\(\s*([^,)]+)\s*\)$')
        
        # Original patterns for other functions that accept 2 arguments  
        self.quoted_function_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?\s*\)$')
        self.function_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,)]+)(?:\s*,\s*([^)]+))?\s*\)$')

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        if not responses:
            # Handle empty list case
            return torch.empty((0, 0), dtype=torch.long)
        
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )


        def clean_response(resp: str) -> str:
            # First, remove any conversation markers if present
            if '<|im_start|>assistant' in resp:
                assistant_idx = resp.find('<|im_start|>assistant')
                if assistant_idx >= 0:
                    after_assistant = resp[assistant_idx:].find('\n')
                    if after_assistant >= 0:
                        resp = resp[assistant_idx + after_assistant + 1:]
                    else:
                        resp = resp[assistant_idx + len('<|im_start|>assistant'):]
            
            # --- MODIFIED LOGIC START ---

            # ORIGINAL LOGIC - Find first occurrence of closing tags
            try:
                answer_idx = resp.index('</answer>')
            except ValueError:
                answer_idx = -1

            try:
                kg_query_idx = resp.index('</kg-query>')
            except ValueError:
                kg_query_idx = -1

            # Determine which tag comes first, if any
            if answer_idx != -1 and (answer_idx < kg_query_idx or kg_query_idx == -1):
                # '</answer>' is present and comes before '</kg-query>' or '</kg-query>' is not present
                return resp[:answer_idx + len('</answer>')]
            elif kg_query_idx != -1:
                # '</kg-query>' is present and comes before '</answer>' or '</answer>' is not present
                return resp[:kg_query_idx + len('</kg-query>')]
            else:
                # No complete tags found, return as-is
                return resp
        
        responses_str = [clean_response(resp) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
        
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        # --- Added safeguard to preserve closing </information> tag after truncation ---
        # Decode each truncated observation string and ensure that any <information> block
        # still ends with a proper closing tag. If the tag is missing (likely due to
        # truncation) we append an ellipsis and the closing tag to avoid malformed
        # markup that confuses the LLM.
        obs_fixed = []
        needs_fix = False
        for obs_ids in next_obs_ids:
            text = self.tokenizer.decode(obs_ids, skip_special_tokens=True)
            if "<information>" in text and "</information>" not in text:
                needs_fix = True
                obs_fixed.append(text.rstrip() + " …</information>")
            else:
                obs_fixed.append(text)
        if needs_fix:
            # Re-tokenize the fixed observations so shapes stay consistent
            next_obs_ids = self.tokenizer(
                obs_fixed,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids']
        # --- End safeguard ---

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Ensure all tensors are on the same device before concatenation
        device = rollings.batch['input_ids'].device # Assume initial device is the target
        
        current_input_ids = rollings.batch['input_ids'].to(device)
        current_responses = cur_responses.to(device)
        current_next_obs_ids = next_obs_ids.to(device)

        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            current_input_ids,
            current_responses,
            current_next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, sorted_indices

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          turn_idx: int = None,
                          active_mask: torch.Tensor = None) -> Dict:
        """Update right side state."""
        device = right_side['responses'].device # Assume initial device is the target
        
        # Debug storage removed - no longer needed after successful fix
        
        current_right_side_responses = right_side['responses'].to(device)
        current_right_side_responses_with_info_mask = right_side['responses_with_info_mask'].to(device)
        current_right_side_turn_tokens = right_side['turn_tokens'].to(device)
        current_cur_responses = cur_responses.to(device)

        if next_obs_ids is not None:
            current_next_obs_ids = next_obs_ids.to(device)
            responses, responses_with_info_mask, sorted_indices = self._info_masked_concatenate_with_padding(
                    current_right_side_responses,
                    current_right_side_responses_with_info_mask,
                    current_cur_responses,
                    current_next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, sorted_indices = self._info_masked_concatenate_with_padding(
                    current_right_side_responses,
                    current_right_side_responses_with_info_mask,
                    current_cur_responses,
                    pad_to_left=False
                )
        # Handle turn token generation
        if turn_idx is not None:
            # Start with -1 for all tokens (no advantage)
            action_turn_tokens = torch.full_like(current_cur_responses, -1, dtype=torch.long, device=device)
            
            # Only assign turn numbers to non-pad tokens within active examples
            if active_mask is not None:
                # Expand active_mask to match response dimensions
                if active_mask.dim() == 1:
                    expanded_active_mask = active_mask.unsqueeze(1).expand_as(current_cur_responses)
                else:
                    expanded_active_mask = active_mask
                
                # Combine active mask with non-pad mask: only assign to real response tokens
                valid_response_mask = expanded_active_mask & (current_cur_responses != self.tokenizer.pad_token_id)
                action_turn_tokens[valid_response_mask] = turn_idx
            else:
                # Fallback: assign turn numbers to all response tokens (old behavior)
                action_turn_tokens.fill_(turn_idx)
            
            if next_obs_ids is not None:
                # Info tokens always get -1 (no advantage)
                info_turn_tokens = torch.full_like(current_next_obs_ids, -1, dtype=torch.long, device=device)
                # Concatenate turn tokens: [action_tokens, info_tokens]
                current_turn_tokens = torch.cat([action_turn_tokens, info_turn_tokens], dim=1)
            else:
                current_turn_tokens = action_turn_tokens
                
            # Update turn token sequence
            turn_tokens = torch.cat([current_right_side_turn_tokens, current_turn_tokens], dim=1)
            
            # CRITICAL FIX: Apply the same reordering to turn tokens as was applied to the actual tokens
            # This ensures turn IDs stay aligned with their corresponding tokens after padding reordering
            turn_tokens = turn_tokens.gather(1, sorted_indices)
        else:
            # Fallback: use existing turn tokens
            turn_tokens = current_right_side_turn_tokens
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len], 
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
            'turn_tokens': turn_tokens[:, :max_len]
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] if padding_size > 0 else v for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size] if padding_size > 0 else v
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        # CRITICAL FIX: Use the actual expanded batch size, not the original
        # When using rollouts (e.g., grpo_rollout_n=8), the batch is already expanded
        actual_batch_size = gen_batch.batch['input_ids'].shape[0]
        
        # Track consecutive format errors per sample (for error loop prevention)
        consecutive_format_errors = [0] * actual_batch_size
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []], 
            'responses_with_info_mask': initial_input_ids[:, []],
            'turn_tokens': initial_input_ids[:, []]
        }
        
        active_mask = torch.ones(actual_batch_size, dtype=torch.bool)
        turns_stats = torch.ones(actual_batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(actual_batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(actual_batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Store detailed interaction information for reward calculation
        # CRITICAL FIX: Use actual_batch_size to match the expanded batch
        # Extract dataset information from gen_batch to include in interaction_history
        dataset_names = None
        if hasattr(gen_batch, 'meta_info') and 'dataset_names' in gen_batch.meta_info:
            dataset_names = gen_batch.meta_info['dataset_names']
        elif hasattr(gen_batch, 'non_tensor_batch') and 'data_source' in gen_batch.non_tensor_batch:
            # Fallback: extract from non_tensor_batch
            dataset_names = gen_batch.non_tensor_batch['data_source']
            if hasattr(dataset_names, 'tolist'):
                dataset_names = dataset_names.tolist()
        
        interaction_history = []
        for i in range(actual_batch_size):
            # Include data_source in each sample's interaction history for MultiTQ detection
            sample_data_source = ''
            if dataset_names and i < len(dataset_names):
                sample_data_source = str(dataset_names[i])
            
            interaction_history.append({
                "data_source": sample_data_source,  # ADD: Include dataset info for reward calculation
                "actions": [],
                "search_results": [],
                "valid_actions": [],
                "is_search_actions": [],
                "raw_server_responses": [],
                "responses_str": [],
                "reasonings": [],
            })
        
        print(f"[DEBUG-INTERACTION] Created interaction_history with data_source info: {[h['data_source'] for h in interaction_history[:3]]}...")

        # Main generation loop - enforce max_turns limit to prevent infinite loops
        # The loop will continue until:
        # 1. All samples are done (active_mask.sum() == 0), OR
        # 2. max_turns is reached (then force final answer)
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            active_count = active_mask.sum().item()
            print(f"[DEBUG-GENERATION] Step {step+1}/{self.config.max_turns}: {active_count}/{actual_batch_size} samples active")
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            print(f"[DEBUG-GENERATION] Step {step+1}: Calling _generate_with_gpu_padding for {active_count} samples...")
            gen_output = self._generate_with_gpu_padding(rollings_active)
            print(f"[DEBUG-GENERATION] Step {step+1}: Generation completed")

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute in environment and process observations
            # Note: The batch is already expanded on the trainer side for multiple rollouts
            # When max_calls is reached, execute_predictions will return FORCE_ANSWER_PROMPT in next_obs
            # and the model should respond with <answer> tag, which will set done=True
            print(f"[DEBUG-GENERATION] Step {step+1}: Calling execute_predictions for {active_count} samples...")
            next_obs, dones, valid_action, is_search, raw_server_responses = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, gen_batch.meta_info, active_mask,
                consecutive_format_errors=consecutive_format_errors
            )
            print(f"[DEBUG-GENERATION] Step {step+1}: execute_predictions completed, {sum(dones)}/{active_count} samples done")
            
            # Store interaction details for this turn
            cur_actions, _, cur_reasonings = self.postprocess_predictions(responses_str)
            for i in range(actual_batch_size):
                if active_mask[i]:  # Active samples: record actual data
                    reasoning_entry = cur_reasonings[i] if i < len(cur_reasonings) else ""
                    interaction_history[i]["actions"].append(cur_actions[i])
                    interaction_history[i]["search_results"].append(next_obs[i])
                    interaction_history[i]["valid_actions"].append(valid_action[i])
                    interaction_history[i]["is_search_actions"].append(is_search[i])
                    interaction_history[i]["responses_str"].append(responses_str[i])
                    interaction_history[i]["reasonings"].append(reasoning_entry)
                else:  # Inactive samples: record placeholder data for consistency
                    interaction_history[i]["actions"].append("")  # No action
                    interaction_history[i]["search_results"].append("")  # No search result
                    interaction_history[i]["valid_actions"].append(0)  # Invalid action
                    interaction_history[i]["is_search_actions"].append(0)  # Not search
                    interaction_history[i]["responses_str"].append("")  # No response
                    interaction_history[i]["reasonings"].append("")
                
                # ALL samples get raw_server_responses recorded for batch consistency
                interaction_history[i]["raw_server_responses"].append(raw_server_responses[i])
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            
            # With expanded batch from trainer side, active_mask should match the expanded size
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                turn_idx=step,
                active_mask=active_mask
            )
        
        # Final turn: if any samples are still active after max_turns, force them to answer
        # This prevents infinite loops when models don't output <answer> or don't query
        if active_mask.sum():
            active_count = active_mask.sum().item()
            print(f"[DEBUG-GENERATION] Final turn after {self.config.max_turns} turns: {active_count}/{actual_batch_size} samples still active, forcing answer...")
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Final turn: do_search=False to prevent further queries, force answer
            next_obs, dones, valid_action, is_search, raw_server_responses = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, gen_batch.meta_info, active_mask, do_search=False,
                consecutive_format_errors=consecutive_format_errors
            )
            
            # Store final turn interaction details
            cur_actions, _, cur_reasonings = self.postprocess_predictions(responses_str)
            for i in range(actual_batch_size):
                if active_mask[i]:
                    reasoning_entry = cur_reasonings[i] if i < len(cur_reasonings) else ""
                    interaction_history[i]["actions"].append(cur_actions[i])
                    interaction_history[i]["search_results"].append(next_obs[i])
                    interaction_history[i]["valid_actions"].append(valid_action[i])
                    interaction_history[i]["is_search_actions"].append(is_search[i])
                    interaction_history[i]["responses_str"].append(responses_str[i])
                    interaction_history[i]["reasonings"].append(reasoning_entry)
                    interaction_history[i]["raw_server_responses"].append(raw_server_responses[i])
                else:
                    interaction_history[i]["actions"].append("")
                    interaction_history[i]["search_results"].append("")
                    interaction_history[i]["valid_actions"].append(0)
                    interaction_history[i]["is_search_actions"].append(0)
                    interaction_history[i]["responses_str"].append("")
                    interaction_history[i]["reasonings"].append("")
                    interaction_history[i]["raw_server_responses"].append({})
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            next_obs_ids = self._process_next_obs(next_obs)
            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                turn_idx=self.config.max_turns,
                active_mask=active_mask
            )
            
            print(f"[DEBUG-GENERATION] Final turn completed: {sum(dones)}/{active_count} samples done")
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        # Add detailed interaction information for reward calculation
        meta_info['interaction_history'] = interaction_history
        
        # Ensure the final output maintains the original batch size
        # This is crucial for GRPO which expands the batch size
        assert original_left_side['input_ids'].shape[0] == actual_batch_size, \
            f"Left side batch size mismatch: {original_left_side['input_ids'].shape[0]} != {actual_batch_size}"
        assert original_right_side['responses'].shape[0] == actual_batch_size, \
            f"Right side batch size mismatch: {original_right_side['responses'].shape[0]} != {actual_batch_size}"
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        
        # Final check to ensure the output has the correct batch size
        assert final_output.batch['responses'].shape[0] == actual_batch_size, \
            f"Final output batch size mismatch: {final_output.batch['responses'].shape[0]} != {actual_batch_size}"
        
        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        # Ensure all tensors are on the same device before concatenation
        device = left_side['input_ids'].device
        
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'].to(device).long(),
            right_side['responses'].to(device).long()
        ], dim=1)
        
        # Ensure responses tensors are long type for tokenizer compatibility
        final_output['responses'] = final_output['responses'].to(device).long()
        final_output['responses_with_info_mask'] = final_output['responses_with_info_mask'].to(device).long()
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'].to(device)),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'].to(device)),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        # Generate turn sequence tensor for multi-turn advantage calculation
        # Create turn tokens for prompt (left_side) - all get -1 (no advantage)
        device = left_side['input_ids'].device
        prompt_turn_tokens = torch.full_like(left_side['input_ids'], -1, dtype=torch.long, device=device)
        
        # Combine prompt and response turn tokens
        final_output['turn_sequence_tensor'] = torch.cat([
            prompt_turn_tokens,
            right_side['turn_tokens']
        ], dim=1)
        
        # DEBUG: LLM token validation (can be enabled for debugging)
        # Note: Successfully fixed turn token alignment - 0 errors achieved!
        # Uncomment below for debugging if needed in the future
        
        # turn_tokens = final_output['turn_sequence_tensor']
        # info_mask = final_output['info_mask']
        # 
        # if turn_tokens.size(0) > 0:
        #     sample_turn = turn_tokens[0]
        #     sample_info = info_mask[0]
        #     llm_generated_positions = (sample_turn > 0)
        #     valid_token_positions = (sample_info == 1)
        #     llm_tokens_with_invalid_mask = llm_generated_positions & ~valid_token_positions
        #     
        #     if llm_tokens_with_invalid_mask.sum() > 0:
        #         print(f"[DEBUG] LLM Token Validation Issues: {llm_tokens_with_invalid_mask.sum().item()} errors found")
        #     else:
        #         print(f"[DEBUG] LLM Token Validation: Perfect alignment achieved!")
        
        # Store the full concatenated sequence for debugging/logging if needed
        # This is what the test script expects as 'full_interaction_ids'
        # It should be a list of tensors, one per batch item.
        # In this case, batch size is implicitly handled by the structure of left_side and right_side.
        # final_output['input_ids'] here is the full sequence for all items in the batch.
        # The test script expects a list containing this tensor.
        meta_info['full_interaction_ids'] = [final_output['input_ids'].clone()] 

        # Ensure meta_info from gen_batch is preserved if it was updated
        # The meta_info passed here is from the last gen_output, might not be the original one.
        # However, sample_id and dataset_name should be stable.
        # For safety, if we need the original gen_batch.meta_info, it should be explicitly carried.
        # For now, assume meta_info contains what's needed or what was last updated.

        final_output_proto = DataProto.from_dict(final_output)
        final_output_proto.meta_info.update(meta_info)
        
        return final_output_proto # Return DataProto object

    def execute_predictions(self, predictions: List[str], pad_token: str, meta_info: Dict, active_mask=None, do_search=True, consecutive_format_errors=None) -> Tuple[List[str], List[bool], List[int], List[int], List[Dict]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            meta_info: Dict containing batch-level metadata OR per-sample metadata
            consecutive_format_errors: List[int] tracking consecutive format errors per sample
            
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search, raw_server_responses)
        """
        if meta_info is None:
            meta_info = {}
        
        # Initialize consecutive_format_errors if not provided (for backward compatibility)
        if consecutive_format_errors is None:
            consecutive_format_errors = [0] * len(predictions)

        cur_actions, contents, _ = self.postprocess_predictions(predictions)
        session_keys = self._build_bridge_session_keys(meta_info, len(cur_actions)) if self.use_sparql_bridge else []
        
        # Check for responses that exceed max_response_length and mark them as invalid
        # Note: In kgqa_agent mode (use_sparql_bridge=True), we don't check response length
        # to match kgqa_agent's behavior (it only uses max_tokens from model config)
        response_length_exceeded = []
        if not self.use_sparql_bridge:
            # Only check response length in non-kgqa_agent mode
            for i, prediction in enumerate(predictions):
                # Tokenize the response to check its length
                response_tokens = self.tokenizer(
                    prediction, 
                    add_special_tokens=False, 
                    return_tensors='pt'
                )['input_ids']
                response_length = response_tokens.shape[1]
                
                if response_length > self.config.max_response_length:
                    response_length_exceeded.append(True)
                    # Override action to be invalid if response is too long
                    cur_actions[i] = 'error'
                    contents[i] = f'Response too long ({response_length} > {self.config.max_response_length} tokens)'
                else:
                    response_length_exceeded.append(False)
        else:
            # In kgqa_agent mode, don't check response length (match kgqa_agent behavior)
            response_length_exceeded = [False] * len(predictions)
        
        next_obs, dones, valid_action, is_search = [], [], [], []
        raw_server_responses = []
        
        # Handle kg-query actions (send to KG server)
        # CRITICAL FIX: Track which sample index each kg-query belongs to
        kg_queries_with_indices = [(i, content) for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'kg-query']
        kg_queries_contents = [content for _, content in kg_queries_with_indices]
        kg_query_indices = [i for i, _ in kg_queries_with_indices]
        
        if do_search and kg_queries_contents:
            # Create per-sample meta_info for KG queries
            # Pass the batch indices of queries to properly extract meta_info
            kg_query_meta_infos = self._extract_kg_meta_info_for_queries(cur_actions, meta_info, active_mask, kg_query_indices)
            kg_results, raw_kg_responses = self.batch_search(kg_queries_contents, kg_query_meta_infos) # Pass per-sample meta_info
            
            # VALIDATION: Ensure we got the right number of responses
            if len(kg_results) != len(kg_queries_contents):
                raise ValueError(
                    f"KG response count mismatch: got {len(kg_results)} results for "
                    f"{len(kg_queries_contents)} queries"
                )
            if len(raw_kg_responses) != len(kg_queries_contents):
                raise ValueError(
                    f"Raw KG response count mismatch: got {len(raw_kg_responses)} responses for "
                    f"{len(kg_queries_contents)} queries"
                )
        else:
            kg_results = [''] * len(kg_queries_contents)
            raw_kg_responses = [{} for _ in kg_queries_contents] # Return empty dicts for no search
        
        # Create a mapping from sample index to kg response
        kg_response_map = {}
        for idx, (sample_idx, _) in enumerate(kg_queries_with_indices):
            kg_response_map[sample_idx] = (kg_results[idx], raw_kg_responses[idx])
        
        # Note: <search> tag is not a valid tag in kgqa_agent mode, removed
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                raw_server_responses.append({})
            else:
                if action == 'answer':
                    # Reset consecutive format errors on successful action
                    consecutive_format_errors[i] = 0
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    raw_server_responses.append({})
                elif action == 'kg-query':
                    # Reset consecutive format errors on successful action
                    consecutive_format_errors[i] = 0
                    # CRITICAL FIX: Use the mapping to get the correct response for this sample
                    if i in kg_response_map:
                        kg_result, raw_kg_response = kg_response_map[i]
                        # Check if max_calls limit was reached (FORCE_ANSWER_PROMPT returned)
                        # When max_calls is reached, we should set done=True to force the model to answer
                        # The FORCE_ANSWER_PROMPT is already in kg_result, so we just need to check the meta
                        is_max_calls_reached = (
                            isinstance(raw_kg_response, dict) and
                            raw_kg_response.get("meta", {}).get("action") == "max_calls_reached"
                        )
                        
                        if is_max_calls_reached:
                            # Max calls reached: use FORCE_ANSWER_PROMPT to force model to answer
                            # Build continuation prompt with the last query results (includes <information> tags)
                            continuation = build_continuation_prompt(kg_result.strip())
                            # Append FORCE_ANSWER_PROMPT after continuation to force answer
                            # Note: continuation already includes <information> and guidance text
                            # Add <think> tag after user message to guide model thinking
                            next_obs.append(f'\n\n{continuation}\n\n{FORCE_ANSWER_PROMPT}\n<think>')
                            raw_server_responses.append(raw_kg_response)
                            dones.append(1)  # Force done to end conversation after model answers
                            valid_action.append(1)  # The query itself was valid
                            is_search.append(1)
                        else:
                            # Normal query: continue conversation using CONTINUATION_PROMPT_TEMPLATE
                            # This includes <information> tags and guidance text
                            continuation = build_continuation_prompt(kg_result.strip())
                            # The continuation prompt already includes <information> and guidance text
                            # Add <think> tag at the end to guide model thinking
                            next_obs.append(f'\n\n{continuation}\n<think>')
                            raw_server_responses.append(raw_kg_response)
                            dones.append(0)  # Continue conversation
                            # Fix: KG queries are invalid if search is disabled (final turn)
                            if do_search:
                                valid_action.append(1)  # Valid when search is enabled
                            else:
                                valid_action.append(0)  # Invalid in final turn when search is disabled
                            is_search.append(1)
                    else:
                        # Fallback if something went wrong
                        # Add <think> tag after error message to guide model thinking
                        next_obs.append(f'\n\n<information>Error: No response found for this query</information>\n\n<think>')
                        raw_server_responses.append({"error": "No response mapped"})
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(1)
                # Note: <search> tag is not a valid tag in kgqa_agent mode, removed
                # If model outputs <search>, it will be treated as 'error' action
                else:
                    # Update consecutive format error count
                    consecutive_format_errors[i] += 1
                    
                    # Check if this is a length-exceeded response
                    if response_length_exceeded[i]:
                        # Add <think> tag after information to guide model thinking
                        next_obs.append(f'\n\n<information>Your previous response was too long ({contents[i]}). Please provide shorter responses within the token limit.</information>\n<think>')
                    elif consecutive_format_errors[i] >= 3:
                        # After 3 consecutive format errors, provide critical guidance (aligned with eval side)
                        # Note: eval side retries 3 times then continues, we provide stronger guidance
                        next_obs.append(
                            f'\n\n<information>CRITICAL: You have made {consecutive_format_errors[i]} '
                            f'consecutive format errors. You MUST output EITHER:\n'
                            f'1. <kg-query>get_relations("entity")</kg-query> to search the knowledge graph, OR\n'
                            f'2. <answer>["Entity1", "Entity2"]</answer> to provide your final answer.\n'
                            f'No other format is accepted. Follow the exact tag format shown above.</information>\n<think>'
                        )
                        # Continue allowing the model to retry (don't force termination yet)
                        # This gives the model one more chance with stronger guidance
                    else:
                        # Add <think> tag after information to guide model thinking
                        next_obs.append(f'\n\n<information>Your previous action is invalid. You should put the query between <kg-query> and </kg-query> if you want to search, or put the answer between <answer> and </answer> if you want to give the final answer.</information>\n<think>')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    raw_server_responses.append({"success": False, "action": "invalid", "kg_metadata": {"success": False, "error_type": KGErrorType.FORMAT_ERROR}})
            
        # CRITICAL FIX: Remove assertion since we're using mapping now
        # Note: search_results removed since <search> tag is not valid in kgqa_agent mode

        if self.use_sparql_bridge and session_keys:
            for idx, done in enumerate(dones):
                if done and idx < len(session_keys):
                    self._reset_bridge_session(session_keys[idx])
            
        return next_obs, dones, valid_action, is_search, raw_server_responses

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str], List[str]]:
        """
        Process (text-based) predictions from llm into actions, their contents, and reasoning traces.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, contents list, reasonings list)
            Content for search is the string "action_type, entity_id [, relation_name]"
            Content for answer is the answer string.
        """
        actions: List[str] = []
        contents: List[str] = []
        reasonings: List[str] = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # Extract reasoning/think content (if present)
                # We manually add <think> tag after information, so we need to handle:
                # 1. Model outputs complete <think>...</think> tag pair (need to deduplicate)
                # 2. Model outputs only </think> closing tag (opening tag was from our prompt)
                # 3. Model outputs no redacted_reasoning tag at all
                
                reasoning_text = ""
                
                if hasattr(self, "think_pattern"):
                    # First, try to match complete tag pair <think>...</think>
                    complete_match = self.think_pattern.search(prediction)
                    
                    if complete_match:
                        # Model output a complete tag pair - extract content and deduplicate
                        # Since we added <think> in the prompt, the model's output might include
                        # the opening tag again, so we extract the content between tags
                        reasoning_text = complete_match.group(1).strip()
                    else:
                        # Check for closing tag </think> (opening tag was from our prompt)
                        closing_tag_pattern = re.compile(r'</think>', re.IGNORECASE)
                        closing_tag_match = closing_tag_pattern.search(prediction)
                        
                        if closing_tag_match:
                            # Extract text before the closing tag
                            # This is the reasoning content that follows our <think> prompt
                            text_before_closing = prediction[:closing_tag_match.start()].strip()
                            
                            # Remove any duplicate <think> opening tag if model added it
                            # (deduplication: remove opening tag if present)
                            text_before_closing = re.sub(r'^<think>\s*', '', text_before_closing, flags=re.IGNORECASE)
                            
                            # Heuristic: if text before closing tag is reasonable length and doesn't contain other tags,
                            # it's likely reasoning content
                            if text_before_closing and len(text_before_closing) < 500:
                                # Check if it contains other action tags (if so, it's probably not pure reasoning)
                                has_action_tags = bool(re.search(r'<(kg-query|answer|search)', text_before_closing, re.IGNORECASE))
                                if not has_action_tags:
                                    reasoning_text = text_before_closing
                
                reasonings.append(reasoning_text)

                # Determine action based on actual tag order
                candidates: List[Tuple[str, int, str]] = []

                # Use findall to get all matches, then take the last one to align with eval side
                answer_matches = self.answer_pattern.findall(prediction)
                if answer_matches:
                    # Take the last match and find its position
                    last_answer_content = answer_matches[-1].strip()
                    # Find the position of the last match
                    last_match = None
                    for match in self.answer_pattern.finditer(prediction):
                        last_match = match
                    if last_match:
                        candidates.append(("answer", last_match.start(), last_answer_content))

                # For kg-query, also take the last match to be consistent
                kg_query_matches = self.kg_query_pattern1.findall(prediction)
                if kg_query_matches:
                    last_kg_content = kg_query_matches[-1].strip()
                    last_match = None
                    for match in self.kg_query_pattern1.finditer(prediction):
                        last_match = match
                    if last_match:
                        candidates.append(("kg-query", last_match.start(), last_kg_content))
                else:
                    kg_query_matches2 = self.kg_query_pattern2.findall(prediction)
                    if kg_query_matches2:
                        last_kg_content = kg_query_matches2[-1].strip()
                        last_match = None
                        for match in self.kg_query_pattern2.finditer(prediction):
                            last_match = match
                        if last_match:
                            candidates.append(("kg-query", last_match.start(), last_kg_content))

                # Note: <search> tag is not a valid tag in kgqa_agent mode, removed
                # Only match <answer> and <kg-query> tags

                if candidates:
                    action, _, content = min(candidates, key=lambda item: item[1])
                else:
                    action = 'error'
                    content = ''
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content) # content is now the string to be parsed by _batch_search

        return actions, contents, reasonings

    def batch_search(self, search_query_contents: List[str] = None, meta_info_list = None) -> Tuple[List[str], List[Dict[str, Any]]]: # Changed to return both
        """
        Batchified search for queries using the KG retrieval server.
        Args:
            search_query_contents: List of strings, each being function call queries or error messages.
            meta_info_list: List of meta_info dicts (one per query) OR single dict for backward compatibility
        Returns:
            A tuple of (formatted_string_results, raw_server_responses) from the KG server.
        """
        if not search_query_contents:
            return [], []

        # Handle error messages directly (like deprecated <search> tag errors)
        results = []
        valid_queries = []
        valid_meta_infos = []
        error_indices = []
        
        for i, content in enumerate(search_query_contents):
            if content.startswith("ERROR:"):
                # This is an error message, return it directly
                results.append(content)
                error_indices.append(i)
            else:
                # This is a valid query to process
                valid_queries.append(content)
                # Get corresponding meta_info
                if isinstance(meta_info_list, list) and i < len(meta_info_list):
                    valid_meta_infos.append(meta_info_list[i])
                elif isinstance(meta_info_list, dict):
                    # Backward compatibility: single meta_info for all queries
                    valid_meta_infos.append(meta_info_list)
                else:
                    # Fallback
                    valid_meta_infos.append({"sample_id": "unknown", "dataset_name": "unknown"})
        
        raw_server_responses = []
        if valid_queries:
            print(f"[DEBUG-KG-SEARCH] batch_search: Processing {len(valid_queries)} KG queries via SPARQL bridge...")
            if self.use_sparql_bridge:
                kg_server_responses, kg_results = self._batch_search_via_sparql_bridge(
                    valid_queries, valid_meta_infos
                )
                print(f"[DEBUG-KG-SEARCH] batch_search: SPARQL bridge completed, got {len(kg_results)} results")
            else:
                kg_server_responses, kg_results = self._batch_search(valid_queries, valid_meta_infos) # Pass list of meta_infos
            
            # Merge error messages and valid results in correct order
            final_results = []
            final_raw_responses = []
            valid_idx = 0
            for i in range(len(search_query_contents)):
                if i in error_indices:
                    final_results.append(results[error_indices.index(i)])
                    # Create a mock error response for consistency
                    final_raw_responses.append({
                        "success": False,
                        "choices": [{"message": {"content": results[error_indices.index(i)]}}]
                    })
                else:
                    final_results.append(kg_results[valid_idx])
                    final_raw_responses.append(kg_server_responses[valid_idx])
                    valid_idx += 1
            return final_results, final_raw_responses
        else:
            # Create mock error responses for all error messages
            error_responses = []
            for result in results:
                error_responses.append({
                    "success": False,
                    "choices": [{"message": {"content": result}}]
                })
            return results, error_responses

    def _batch_search(self, search_query_contents: List[str], meta_info_list) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Sends a batch of requests to the KG retrieval server.
        Args:
            search_query_contents: List of strings, each being "action_type, entity_id [, relation_name]".
            meta_info_list: List of meta_info dicts, one per query.
        Returns:
            A tuple of (raw_server_responses, formatted_string_responses) from the KG server.
        """
        parsed_requests = []

        for i, query_content_str in enumerate(search_query_contents):
            
            # Get meta_info for this specific query
            if isinstance(meta_info_list, list) and i < len(meta_info_list):
                meta_info = meta_info_list[i]
            elif isinstance(meta_info_list, dict):
                # Backward compatibility
                meta_info = meta_info_list
            else:
                meta_info = {"sample_id": "unknown", "dataset_name": "webqsp"}
            
            # Retrieve sample_id and dataset_name from this query's meta_info
            sample_id = meta_info.get("sample_id", "unknown_sample_id")
            dataset_name = meta_info.get("dataset_name", "webqsp")  # Use webqsp as safe fallback
            
            # If we still have unknown_sample_id, generate a better fallback
            if sample_id == "unknown_sample_id":
                sample_id = f"generated_sample_{i:06d}"
            
            try:
                parsed = self._parse_kg_query(query_content_str)
                action_type = parsed["action_type"]
                entity_id = parsed["entity_id"]
                relation_name = parsed["relation_name"]
                
                # Client-side validation: Check if action_type is valid
                valid_action_types = [
                    "get_relations_in", "get_relations_out", "get_entities_in", "get_entities_out",  # New names
                    "get_relations", "get_head_relations", "get_tail_relations", "get_head_entities", "get_tail_entities"  # Legacy
                ]
                if action_type not in valid_action_types:
                    # Create descriptive error message for invalid action type
                    error_msg = (
                        f"Invalid action type '{action_type}'. "
                        f"Use one of: get_relations_in, get_relations_out, get_entities_in, get_entities_out. "
                        f"Example: get_relations_in(\"entity\") or get_entities_out(\"entity\", \"relation\")"
                    )
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue

                # Client-side validation: Check if entity_id is not empty
                if not entity_id or not entity_id.strip():
                    error_msg = "Entity name cannot be empty. Provide a valid entity name in quotes."
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue

                # Client-side validation: Check relation requirements
                if action_type in ["get_entities_in", "get_entities_out", "get_head_entities", "get_tail_entities"] and (not relation_name or not relation_name.strip()):
                    error_msg = f"Relation name is required for {action_type}. Use format: {action_type}(\"entity\", \"relation\")"
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue
                
                request_payload = {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "action_type": action_type,
                    "entity_id": entity_id.strip(),  # Clean up entity_id
                }
                # Add batch_index for debugging if available
                if "batch_index" in meta_info:
                    request_payload["_debug_batch_idx"] = meta_info["batch_index"]
                # Only add relation to payload if it's not None and not an empty string.
                # The KG server expects the key to be absent if no relation is applicable or provided.
                # If relation_name is an empty string here, it means the LLM provided 3 parts but the 3rd was empty.
                # For actions like get_tail_entities, an empty relation is invalid, and the server should handle that.
                if relation_name and relation_name.strip(): # This will be true if relation_name is a non-empty string
                    request_payload["relation"] = relation_name.strip()
                
                parsed_requests.append(request_payload)

            except Exception as e:
                
                # Create LLM-friendly error message
                error_msg = self._create_query_format_error(query_content_str, str(e))
                
                parsed_requests.append({
                    "_is_client_error": True, 
                    "error_message": error_msg,
                })

        if not parsed_requests:
            return [], []

        # Filter out client-side errors before sending to server
        valid_requests = [req for req in parsed_requests if "_is_client_error" not in req]
        
        server_responses = []
        if valid_requests:
            # NEW: Group requests by dataset for dynamic routing
            requests_by_server = defaultdict(list)
            request_to_server_mapping = {}
            
            for idx, req in enumerate(valid_requests):
                dataset_name = req.get("dataset_name", "")
                server_url = self._get_server_url_for_dataset(dataset_name)
                requests_by_server[server_url].append(req)
                request_to_server_mapping[idx] = server_url
            
            # Send requests to appropriate servers
            server_url_to_responses = {}
            
            for server_url, server_requests in requests_by_server.items():
                try:
                    print(f"[KG_ROUTING] Sending {len(server_requests)} requests to {server_url}")
                    
                    # Add timeout and connection settings for training stability
                    response = requests.post(
                        server_url, 
                        json=server_requests, 
                        timeout=15,  # Unified default timeout
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    server_url_to_responses[server_url] = response.json()
                    
                    print(f"[KG_ROUTING] Received {len(server_url_to_responses[server_url])} responses from {server_url}")
                    
                except requests.exceptions.Timeout as e:
                    # Handle timeout errors specifically  
                    print(f"[KG_TIMEOUT_ERROR] Request timed out to {server_url}")
                    error_responses = []
                    for req in server_requests:
                        error_msg = f"KG server timeout ({server_url})"
                        error_responses.append({
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except requests.exceptions.ConnectionError as e:
                    # Handle connection errors specifically
                    print(f"[KG_CONNECTION_ERROR] Connection failed to {server_url}: {str(e)}")
                    error_responses = []
                    for req in server_requests:
                        error_msg = f"KG server connection failed ({server_url})"
                        error_responses.append({
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except requests.exceptions.RequestException as e:
                    # Handle other HTTP errors with detailed logging
                    print(f"[KG_REQUEST_ERROR] HTTP error to {server_url}: {str(e)}")
                    # Add debug info about the request that failed
                    
                    error_responses = []
                    for req in server_requests:
                        error_msg = self._format_http_error(e, req)
                        error_responses.append({
                            "error": error_msg, 
                            "query_time": 0, 
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except json.JSONDecodeError as e:
                    # Handle invalid JSON responses
                    print(f"[KG_JSON_ERROR] Invalid JSON response from {server_url}: {str(e)}")
                    error_msg = f"KG server response error ({server_url})"
                    error_responses = [
                        {
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        }
                        for req in server_requests
                    ]
                    server_url_to_responses[server_url] = error_responses
            
            # Reconstruct responses in original order
            server_responses = []
            server_response_idx = defaultdict(int)  # Track response index for each server
            
            for idx, req in enumerate(valid_requests):
                server_url = request_to_server_mapping[idx]
                response_idx = server_response_idx[server_url]
                server_responses.append(server_url_to_responses[server_url][response_idx])
                server_response_idx[server_url] += 1

        # Merge client-side errors with server responses in correct order
        final_responses = []
        server_idx = 0
        for req in parsed_requests:
            if "_is_client_error" in req:
                final_responses.append({
                    "error": req["error_message"], 
                    "query_time": 0, 
                    "total_results": 0,
                    "request_payload": req,  # Include original request for unique query tracking
                    "kg_metadata": {"success": False, "error_type": KGErrorType.FORMAT_ERROR},
                })
            else:
                response = server_responses[server_idx].copy()  # Make a copy to avoid modifying original
                response["request_payload"] = req  # Attach the original request payload
                if "kg_metadata" not in response:
                    response["kg_metadata"] = {
                        "success": response.get("success", True),
                        "error_type": response.get("error_type", KGErrorType.SUCCESS)
                    }
                final_responses.append(response)
                server_idx += 1

        # VALIDATION: Ensure response count matches query count
        if len(final_responses) != len(search_query_contents):
            raise ValueError(
                f"Final response count mismatch in _batch_search: "
                f"got {len(final_responses)} responses for {len(search_query_contents)} queries"
            )
        
        formatted_responses = [self._passages2string(item) for item in final_responses]
        
        # VALIDATION: Ensure formatted response count matches
        if len(formatted_responses) != len(search_query_contents):
            raise ValueError(
                f"Formatted response count mismatch in _batch_search: "
                f"got {len(formatted_responses)} formatted responses for {len(search_query_contents)} queries"
            )
        
        return final_responses, formatted_responses

    def _batch_search_via_sparql_bridge(
        self,
        search_query_contents: List[str],
        meta_info_list: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute kgqa-style queries directly against the SPARQL endpoint."""
        if not self.kgqa_adapter:
            raise RuntimeError("SPARQL bridge requested but adapter is not initialized.")

        query_count = len(search_query_contents)
        if query_count == 0:
            return [], []

        print(
            f"[DEBUG-KG-SEARCH] _batch_search_via_sparql_bridge: Processing {query_count} queries "
            f"with up to {self.kgqa_thread_pool_size} threads..."
        )

        import time

        max_workers = max(1, min(self.kgqa_thread_pool_size, query_count))
        raw_payloads: List[Optional[Dict[str, Any]]] = [None] * query_count
        formatted_results: List[Optional[str]] = [None] * query_count

        def _build_error_payload(idx: int, meta_info: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
            original_sample_id = meta_info.get("sample_id", f"sparql_sample_{idx:06d}")
            batch_index = meta_info.get("batch_index", idx)
            session_key = meta_info.get("bridge_session_key") or f"{original_sample_id}__idx_{batch_index}"
            payload = {
                "success": False,
                "error": error_msg,
                "request_payload": {
                    "original_sample_id": original_sample_id,
                    "bridge_session_key": session_key,
                    "query_index": idx,
                    "query_str": search_query_contents[idx],
                },
                "meta": {
                    "action": "sparql_exception",
                    "error_type": KGErrorType.SERVER_ERROR,
                },
                "kg_metadata": {
                    "success": False,
                    "error_type": KGErrorType.SERVER_ERROR,
                },
            }
            return payload

        def _run_single(idx: int) -> Tuple[int, Dict[str, Any], str]:
            query_content_str = search_query_contents[idx]
            meta_info = meta_info_list[idx] if idx < len(meta_info_list) else {}
            original_sample_id = meta_info.get("sample_id", f"sparql_sample_{idx:06d}")
            batch_index = meta_info.get("batch_index", idx)
            session_key = meta_info.get("bridge_session_key") or f"{original_sample_id}__idx_{batch_index}"
            question_text = meta_info.get("question_text", "")
            topic_entities = meta_info.get("topic_entities")

            query_start_time = time.time()
            print(
                f"[DEBUG-KG-SEARCH] (threaded) Processing query {idx+1}/{query_count}: "
                f"{query_content_str[:100]}..."
            )
            try:
                formatted, payload = self.kgqa_adapter.run_query(
                    sample_id=session_key,
                    query_str=query_content_str,
                    question=question_text,
                    topic_entities=topic_entities,
                )
                query_elapsed = time.time() - query_start_time
                # Log slow queries for analysis
                if query_elapsed > 30:
                    print(
                        f"[DEBUG-KG-SEARCH] (threaded) Query {idx+1}/{query_count} completed in {query_elapsed:.2f}s (SLOW - may indicate complex SPARQL queries)"
                    )
                else:
                    print(
                        f"[DEBUG-KG-SEARCH] (threaded) Query {idx+1}/{query_count} completed in {query_elapsed:.2f}s"
                    )
            except Exception as exc:
                query_elapsed = time.time() - query_start_time
                error_msg = f"SPARQL error after {query_elapsed:.2f}s: {exc}"
                print(
                    f"[DEBUG-KG-SEARCH] (threaded) Query {idx+1}/{query_count} FAILED: {error_msg}"
                )
                payload = _build_error_payload(idx, meta_info, error_msg)
                formatted = "No relations found."
                return idx, payload, formatted

            payload.setdefault("request_payload", {})
            payload["request_payload"].setdefault("original_sample_id", original_sample_id)
            payload["request_payload"].setdefault("bridge_session_key", session_key)
            payload["request_payload"].setdefault("query_index", idx)
            return idx, payload, formatted

        # Remove timeout on future.result() to allow successful queries to complete
        # SPARQLWrapper's internal timeout (15s per query) will still prevent individual queries from hanging
        # This allows get_triples with multiple relations to complete even if total time exceeds 15s
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single, idx): idx for idx in range(query_count)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # No timeout - allow successful queries to complete regardless of duration
                    result_idx, payload, formatted = future.result()
                except Exception as exc:
                    meta_info = meta_info_list[idx] if idx < len(meta_info_list) else {}
                    error_msg = f"SPARQL future failed: {exc}"
                    print(f"[DEBUG-KG-SEARCH] (threaded) Query {idx+1}/{query_count} EXCEPTION: {error_msg}")
                    payload = _build_error_payload(idx, meta_info, error_msg)
                    formatted = "No relations found."
                    result_idx = idx

                raw_payloads[result_idx] = payload
                formatted_results[result_idx] = formatted

        # Ensure no None remains (should not happen, but guard to avoid crashes)
        for idx in range(query_count):
            if raw_payloads[idx] is None or formatted_results[idx] is None:
                meta_info = meta_info_list[idx] if idx < len(meta_info_list) else {}
                fallback_msg = "SPARQL query returned no result"
                raw_payloads[idx] = _build_error_payload(idx, meta_info, fallback_msg)
                formatted_results[idx] = "No relations found."  # Simple message like kgqa_agent

        return raw_payloads, formatted_results  # type: ignore

    def _passages2string(self, kg_server_response_item: Dict[str, Any]) -> str:
        """
        Formats a single KG server response item into a string for the LLM.
        Updated to handle the new kg_retrieval response format.
        """
        if not isinstance(kg_server_response_item, dict):
            return f"Error: Unexpected KG server response format: {type(kg_server_response_item)}"

        # Handle the new kg_retrieval format
        if "object" in kg_server_response_item and kg_server_response_item.get("object") == "kg_retrieval":
            # New format: kg_retrieval response
            if not kg_server_response_item.get("success", False):
                # Error case - return the error message directly without prefixing
                if "choices" in kg_server_response_item and kg_server_response_item["choices"]:
                    error_content = kg_server_response_item["choices"][0].get("message", {}).get("content", "Unknown error")
                    return error_content
                return "Unknown error occurred. Please check your search format."
            
            # Success case - extract content from choices
            if "choices" in kg_server_response_item and kg_server_response_item["choices"]:
                content = kg_server_response_item["choices"][0].get("message", {}).get("content", "No content")
                return content
            
            return "No content in KG server response"
        
        # Handle old format for backward compatibility
        if "error" in kg_server_response_item:
            return kg_server_response_item['error']  # Return error message directly

        action_actual_results = kg_server_response_item.get("results")
        if not action_actual_results or not isinstance(action_actual_results, list) or len(action_actual_results) == 0:
            # This case might also indicate an error handled by the server and put into the 'results' list.
            # e.g. {"results": [{"error": "subgraph not found"}]}
            if isinstance(action_actual_results, list) and len(action_actual_results) > 0 and "error" in action_actual_results[0]:
                 return action_actual_results[0]['error']  # Return error message directly
            return "No results found for your search query."

        data_payload = action_actual_results[0] # The actual data is wrapped in a list

        if "error" in data_payload: # Error specific to the KG operation for that item
            return data_payload['error']  # Return error message directly

        if "relations" in data_payload:
            relations = data_payload['relations']
            if relations:
                return f"Found relations: {', '.join(relations)}."
            else:
                return "No relations found."
        elif "head_entities" in data_payload:
            head_entities = data_payload['head_entities']
            if head_entities:
                return f"Found head entities: {', '.join(head_entities)}."
            else:
                return "No head entities found."
        elif "tail_entities" in data_payload:
            tail_entities = data_payload['tail_entities']
            if tail_entities:
                return f"Found tail entities: {', '.join(tail_entities)}."
            else:
                return "No tail entities found."
        else:
            return f"Retrieved data: {json.dumps(data_payload)}" # Fallback for other structures

    def _get_available_actions_from_server(self) -> List[str]:
        """Get available actions from the server endpoint."""
        try:
            response = requests.get(f"{self.config.search_url.rstrip('/retrieve')}/actions")
            if response.status_code == 200:
                return response.json().get("actions", [])
        except Exception:
            pass
        # Fallback to default actions (new naming scheme)
        return ["get_relations_in", "get_relations_out", "get_entities_in", "get_entities_out"]

    def _create_query_format_error(self, query_content: str, original_error: str) -> str:
        """Create an LLM-friendly error message for query format issues."""
        
        # For kgqa_agent mode (SPARQL bridge), use get_relations and get_triples
        if self.use_sparql_bridge:
            return (
                f"Invalid query format: '{query_content}'. "
                f"Use function call format like: get_relations(\"entity\") or get_triples(\"entity\", [\"relation1\", \"relation2\"]). "
                f"Available functions: get_relations(\"entity\"), get_triples(\"entity\", [\"relation1\", \"relation2\"]). "
            )
        
        # For kg-r1 mode (FastAPI), get available actions dynamically from server
        available_actions = self._get_available_actions_from_server()
        functions_list = ", ".join(available_actions)
        
        return f"Query format error. Available functions: {functions_list}"

    def _parse_kg_query(self, query_content_str: str) -> Dict[str, Any]:
        """
        Parse KG query in function-call format.
        
        Format: "action_type("entity_name"[, "relation_name"])" (quoted format - preferred)
        Also accepts: "action_type(entity_name[, relation_name])" (unquoted format for backward compatibility)
        
        Returns:
            Dictionary with parsed components: action_type, entity_id, relation_name
        """
        
        query_content_str = query_content_str.strip()
        
        # Try to strip known problematic prefixes using pre-compiled patterns
        cleaned_query = query_content_str
        for prefix_pattern in self.prefix_patterns:
            cleaned_query = prefix_pattern.sub('', cleaned_query).strip()
            
        # Also handle cases like 'kg-query execute "get_head_entities(...)"' 
        # where the function call is wrapped in extra quotes
        if cleaned_query.startswith('"') and cleaned_query.endswith('"'):
            cleaned_query = cleaned_query[1:-1].strip()
        elif cleaned_query.startswith("'") and cleaned_query.endswith("'"):
            cleaned_query = cleaned_query[1:-1].strip()
            
        # Special case: handle malformed quotes like 'get_head_entities(...))"'
        if cleaned_query.endswith(')"') and not cleaned_query.startswith('"'):
            cleaned_query = cleaned_query[:-1]  # Remove trailing quote
        elif cleaned_query.endswith(")'") and not cleaned_query.startswith("'"):
            cleaned_query = cleaned_query[:-1]  # Remove trailing quote
            
        # Handle nested function calls using pre-compiled pattern
        nested_match = self.nested_pattern.match(cleaned_query)
        if nested_match:
            cleaned_query = nested_match.group(1).strip()
        
        # Check for get_relations, get_head_relations, get_tail_relations first - strict single entity only
        get_relations_quoted_match = self.get_relations_quoted_pattern.match(cleaned_query)
        if get_relations_quoted_match:
            action_type = get_relations_quoted_match.group(1)
            entity_id = get_relations_quoted_match.group(2)
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": None
            }
            
        get_relations_unquoted_match = self.get_relations_unquoted_pattern.match(cleaned_query)
        if get_relations_unquoted_match:
            action_type = get_relations_unquoted_match.group(1)
            entity_id = get_relations_unquoted_match.group(2).strip()
            # Strip quotes if present
            if entity_id.startswith('"') and entity_id.endswith('"'):
                entity_id = entity_id[1:-1]
            elif entity_id.startswith("'") and entity_id.endswith("'"):
                entity_id = entity_id[1:-1]
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": None
            }
        
        # Try function-call format with quotes for other functions (get_head_entities, get_tail_entities)
        quoted_match = self.quoted_function_pattern.match(cleaned_query)
        
        if quoted_match:
            action_type = quoted_match.group(1)
            
            # Reject single-entity functions with 2+ arguments - should have been caught above
            if action_type in ["get_relations", "get_relations_in", "get_relations_out", "get_head_relations", "get_tail_relations"]:
                if self.use_sparql_bridge:
                    # kgqa_agent mode: use get_relations and get_triples
                    error_msg = (
                        f"Invalid {action_type} format: '{query_content_str}'. "
                        f"{action_type} accepts only one entity argument: {action_type}(\"entity\"). "
                        f"For relation-specific queries, use get_triples(\"entity\", [\"relation1\", \"relation2\"])."
                    )
                else:
                    # kg-r1 mode: use get_entities_out/get_entities_in
                    error_msg = (
                        f"Invalid {action_type} format: '{query_content_str}'. "
                        f"{action_type} accepts only one entity argument: {action_type}(\"entity\"). "
                        f"For relation-specific queries, use get_entities_out(\"entity\", \"relation\") or get_entities_in(\"entity\", \"relation\")."
                    )
                raise ValueError(error_msg)
            
            entity_id = quoted_match.group(2)  # Quotes are automatically removed by the regex groups
            relation_name = quoted_match.group(3) if quoted_match.group(3) else None
            
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": relation_name
            }
        
        # Try function-call format without quotes for other functions (get_head_entities, get_tail_entities)
        match = self.function_pattern.match(cleaned_query)
        
        if match:
            action_type = match.group(1).strip()
            
            # Reject single-entity functions with 2+ arguments - should have been caught above
            if action_type in ["get_relations", "get_relations_in", "get_relations_out", "get_head_relations", "get_tail_relations"]:
                if self.use_sparql_bridge:
                    # kgqa_agent mode: use get_relations and get_triples
                    error_msg = (
                        f"Invalid {action_type} format: '{query_content_str}'. "
                        f"{action_type} accepts only one entity argument: {action_type}(\"entity\") or {action_type}(entity). "
                        f"For relation-specific queries, use get_triples(\"entity\", [\"relation1\", \"relation2\"])."
                    )
                else:
                    # kg-r1 mode: use get_entities_out/get_entities_in
                    error_msg = (
                        f"Invalid {action_type} format: '{query_content_str}'. "
                        f"{action_type} accepts only one entity argument: {action_type}(\"entity\") or {action_type}(entity). "
                        f"For relation-specific queries, use get_entities_out(\"entity\", \"relation\") or get_entities_in(\"entity\", \"relation\")."
                    )
                raise ValueError(error_msg)
            
            entity_id = match.group(2).strip()
            relation_name = match.group(3).strip() if match.group(3) else None
            
            # CRITICAL FIX: Strip quotes from entity_id and relation_name when using unquoted pattern
            # This handles mixed quoting cases like: get_tail_entities(m.046vpjr, "relation_name")
            if entity_id.startswith('"') and entity_id.endswith('"'):
                entity_id = entity_id[1:-1]
            elif entity_id.startswith("'") and entity_id.endswith("'"):
                entity_id = entity_id[1:-1]
                
            if relation_name:
                if relation_name.startswith('"') and relation_name.endswith('"'):
                    relation_name = relation_name[1:-1]
                elif relation_name.startswith("'") and relation_name.endswith("'"):
                    relation_name = relation_name[1:-1]
            
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": relation_name
            }
        
        # No valid format found - provide clear error message with quoted format examples
        if self.use_sparql_bridge:
            # kgqa_agent mode: use get_relations and get_triples
            error_msg = (
                f"Invalid query format: '{query_content_str}'. "
                f"Use function call format like: get_relations(\"entity\") or get_triples(\"entity\", [\"relation1\", \"relation2\"]). "
                f"Available functions: get_relations(\"entity\"), get_triples(\"entity\", [\"relation1\", \"relation2\"]). "
                f"Example: get_relations(\"Barack Obama\")"
            )
        else:
            # kg-r1 mode: use get_relations_in/get_relations_out/get_entities_in/get_entities_out
            error_msg = (
                f"Invalid query format: '{query_content_str}'. "
                f"Use function call format like: get_relations_in(\"entity\") or get_entities_out(\"entity\", \"relation\"). "
                f"Available functions: get_relations_in(\"entity\"), get_relations_out(\"entity\"), get_entities_out(\"entity\", \"relation\"), get_entities_in(\"entity\", \"relation\"). "
                f"Example: get_relations_in(\"Natalie Portman\")"
            )
        raise ValueError(error_msg)

    def _extract_kg_meta_info_for_queries(self, cur_actions: List[str], meta_info: Dict, active_mask, kg_query_batch_indices: List[int] = None) -> List[Dict]:
        """
        Extract per-sample meta_info for KG queries.
        
        Args:
            cur_actions: List of actions for each sample
            meta_info: Batch meta_info that may contain per-sample data
            active_mask: Mask indicating which samples are active
            
        Returns:
            List of meta_info dicts, one for each KG query
        """
        kg_query_meta_infos = []
        bridge_session_keys = meta_info.get("bridge_session_keys", []) if meta_info else []
        question_list = meta_info.get("original_questions", []) if meta_info else []
        
        # Check if we have per-sample information available
        batch_sample_ids = meta_info.get("sample_ids", [])
        batch_dataset_names = meta_info.get("dataset_names", [])
        fallback_sample_id = meta_info.get("sample_id", "unknown_sample_id")
        
        
        # VALIDATION: The trainer must have already expanded sample_ids after DataProto.repeat()
        # This is critical for correct query-response mapping with GRPO rollouts
        if batch_sample_ids:
            if len(batch_sample_ids) != len(cur_actions):
                raise ValueError(
                    f"Sample IDs length mismatch: len(batch_sample_ids)={len(batch_sample_ids)} != "
                    f"len(cur_actions)={len(cur_actions)}. The trainer should have expanded sample_ids "
                    f"after DataProto.repeat() for GRPO rollouts."
                )
        
        # Generate fallback sample IDs only if we don't have per-sample info at all
        if not batch_sample_ids:
            if fallback_sample_id == "unknown_sample_id":
                batch_sample_ids = [f"fallback_sample_{i:06d}" for i in range(len(cur_actions))]
            else:
                # If we have a single fallback_sample_id, it means we're in a non-expanded batch scenario
                batch_sample_ids = [fallback_sample_id] * len(cur_actions)
        
        # Better fallback logic for dataset_name
        fallback_dataset_name = meta_info.get("dataset_name", None)
        if fallback_dataset_name is None:
            # Try to extract from data_source if available
            data_source = meta_info.get("data_source", None)
            if data_source:
                if isinstance(data_source, list) and len(data_source) > 0:
                    data_source = data_source[0]
                
                if data_source in ["webqsp_kg", "webqsp"]:
                    fallback_dataset_name = "webqsp"
                elif data_source in ["cwq_kg", "cwq"]:
                    fallback_dataset_name = "CWQ"
                elif data_source in ["kgR1_multitq", "multitq"]:
                    fallback_dataset_name = "multitq"
                else:
                    fallback_dataset_name = "webqsp"  # Safe fallback
            else:
                fallback_dataset_name = "webqsp"  # Safe fallback instead of "unknown_dataset_name"
        
        # Generate fallback dataset names if we don't have per-sample info
        if not batch_dataset_names:
            batch_dataset_names = [fallback_dataset_name] * len(cur_actions)
        
        # If kg_query_batch_indices is provided, use it directly
        if kg_query_batch_indices:
            for kg_query_idx, batch_idx in enumerate(kg_query_batch_indices):
                # Get per-sample info for this specific batch index
                if batch_idx < len(batch_sample_ids):
                    sample_id = batch_sample_ids[batch_idx]
                else:
                    if fallback_sample_id == "unknown_sample_id":
                        sample_id = f"query_sample_{batch_idx:06d}"  # Generate unique sample ID
                    else:
                        sample_id = fallback_sample_id
                    
                if batch_idx < len(batch_dataset_names):
                    dataset_name = batch_dataset_names[batch_idx]
                else:
                    dataset_name = fallback_dataset_name
                
                session_key = None
                if batch_idx < len(bridge_session_keys):
                    session_key = bridge_session_keys[batch_idx]
                question_text = ""
                if batch_idx < len(question_list):
                    question_text = question_list[batch_idx]

                kg_query_meta_infos.append({
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "batch_index": batch_idx,  # Add absolute batch index for debugging
                    "bridge_session_key": session_key,
                    "question_text": question_text,
                })
        else:
            # Fallback to old behavior if kg_query_batch_indices not provided
            kg_query_idx = 0
            for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
                if active and action == 'kg-query':
                    # Get per-sample info if available, otherwise use fallback
                    if i < len(batch_sample_ids):
                        sample_id = batch_sample_ids[i]
                    else:
                        if fallback_sample_id == "unknown_sample_id":
                            sample_id = f"query_sample_{i:06d}"  # Generate unique sample ID
                        else:
                            sample_id = fallback_sample_id
                        
                    if i < len(batch_dataset_names):
                        dataset_name = batch_dataset_names[i]
                    else:
                        dataset_name = fallback_dataset_name
                    
                    session_key = None
                    if i < len(bridge_session_keys):
                        session_key = bridge_session_keys[i]
                    question_text = ""
                    if i < len(question_list):
                        question_text = question_list[i]
                    
                    kg_query_meta_infos.append({
                        "sample_id": sample_id,
                        "dataset_name": dataset_name,
                        "batch_index": i,  # Add absolute batch index for debugging
                        "bridge_session_key": session_key,
                        "question_text": question_text,
                    })
                    kg_query_idx += 1
        
        return kg_query_meta_infos

    def _format_http_error(self, error: Exception, request_data: Dict = None) -> str:
        """
        Format HTTP transport errors into concise messages.
        Note: Server-side errors are now handled by the KG server with detailed messages.
        This only handles HTTP transport failures.
        
        Args:
            error: The HTTP exception
            request_data: The original request data (used for context in fallback cases)
            
        Returns:
            Concise error message for HTTP transport failures
        """
        error_str = str(error).lower()
        
        # Extract entity name from request if available for fallback
        entity_name = "query"
        if request_data and "entity_id" in request_data:
            entity_name = f"'{request_data['entity_id']}'"
        
        # For actual server errors that contain detailed error messages, try to extract them
        if hasattr(error, 'response') and error.response is not None:
            try:
                # Try to get the detailed server error message
                response_data = error.response.json()
                
                # Handle list response (batch requests)
                if isinstance(response_data, list) and len(response_data) > 0:
                    server_response = response_data[0]
                else:
                    server_response = response_data
                
                # Try multiple extraction strategies
                detailed_error = None
                
                # Strategy 1: kg_retrieval format with choices
                if ("choices" in server_response and 
                    len(server_response["choices"]) > 0 and
                    "message" in server_response["choices"][0] and
                    "content" in server_response["choices"][0]["message"]):
                    detailed_error = server_response["choices"][0]["message"]["content"]
                
                # Strategy 2: Direct error field
                elif "error" in server_response:
                    detailed_error = server_response["error"]
                
                # Strategy 3: Detail field (FastAPI validation errors)
                elif "detail" in server_response:
                    if isinstance(server_response["detail"], list) and len(server_response["detail"]) > 0:
                        # FastAPI validation error format
                        detail_item = server_response["detail"][0]
                        if "msg" in detail_item:
                            detailed_error = detail_item["msg"]
                        else:
                            detailed_error = str(server_response["detail"])
                    else:
                        detailed_error = str(server_response["detail"])
                
                # Strategy 4: Success=False format  
                elif (server_response.get("success") is False and 
                      "message" in server_response):
                    detailed_error = server_response["message"]
                
                if detailed_error:
                    return detailed_error
                
                # If we got a response but couldn't extract error, log it for debugging
                
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError) as parse_error:
                print(f"[KG_PARSE_ERROR] Failed to parse server response: {parse_error}")
                # Try to get raw response text for debugging
                try:
                    response_text = error.response.text[:200] if hasattr(error.response, 'text') else "No response text"
                    print(f"[KG_RAW_RESPONSE] First 200 chars: {response_text}")
                except:
                    pass
        
        # Handle HTTP transport errors with simple, generic messages
        if "timeout" in error_str or "timed out" in error_str:
            return "KG server request timed out"
        elif "connection" in error_str or "cannot connect" in error_str:
            return "Cannot connect to KG server"
        elif "404" in error_str or "not found" in error_str:
            return "KG server endpoint not found"
        elif "500" in error_str or "internal server error" in error_str:
            return "KG server internal error"
        elif "503" in error_str or "service unavailable" in error_str:
            return "KG server unavailable"
        else:
            # Generic fallback for any other HTTP transport issues
            return f"KG server request failed for {entity_name}"


