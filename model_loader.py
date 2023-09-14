import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
import os

from dotenv import load_dotenv
load_dotenv()

import yaml
with open('config.yml', 'r') as file:
    service_parameters = yaml.safe_load(file)


device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'


class ModelLoader:
    def __init__(self):
        self.tokenizer_path = service_parameters['models']['tokenizer_name']
        self.model_path = service_parameters['models']['model_name']
        self.config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, 
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )    
        self.model = self._load_model()
        self.llm = self._setting_pipeline()

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            device_map="auto",
        )

        return model

    def _setting_pipeline(self):
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids, scores) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        pipeline = transformers.pipeline(
            model=self.model, 
            tokenizer=self.tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=stopping_criteria,
            temperature=0.1,  
            max_new_tokens=512,
            repetition_penalty=1.1 
        )

        llm = HuggingFacePipeline(pipeline=pipeline)

        return llm