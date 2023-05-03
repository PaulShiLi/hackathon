import os
from pathlib import Path
import sys
# insert root path
sys.path.insert(1, os.path.join(Path(__file__).resolve().parent.parent.parent))

from typing import List, Optional
from utils.info import PATH
from transformers import LlamaForCausalLM, LlamaPreTrainedModel, LlamaTokenizer
import torch

class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LlamaForCausalLM.from_pretrained(PATH.MODELS, cache_dir = PATH.CACHE, local_files_only = True)
        self.model = self.model.to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(PATH.MODELS, cache_dir = PATH.CACHE, local_files_only = True)
    
    def predict(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        total_tokens: int = 2000,
        temperature: float = 0.75,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0
    ) -> List[str]:
        """_summary_

        Args:
            prompt (str): The prompt for the model to generate from.
            context (Optional[List[str]], optional): Additional context to finetune the prompt answer. Defaults to None.
            total_tokens (int, optional): Maximum tokens to output. Defaults to 2000.
            temperature (float, optional): The temperature of the model to control its randomness in text generation. Defaults to 0.75.
            top_p (float, optional): _description_. Defaults to 1.0.
            repetition_penalty (float, optional): _description_. Defaults to 1.0.

        Returns:
            List[str]: _description_
        """
        context = "" if context == [] else "\n".join(context)
        prompt = f"{context}\n{prompt}"
        prompt = self.tokenizer.encode(prompt, return_tensors = "pt").input_ids.to(self.device)
        
        output = self.model.generate(
            prompt,
            num_beams = 5,
            max_length = total_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
        )
        
        out = self.tokenizer.batch_decode(output[0], skip_special_tokens = True)
        
        return out