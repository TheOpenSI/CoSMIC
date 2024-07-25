import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model_external(llm_model, is_finetune=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if is_finetune:
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            # add_eos_token = True,
            add_bos_token = True
        )

        tokenizer.pad_token = tokenizer.eos_token

        # load model
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", 
            quantization_config=bnb_config, 
            use_cache=True,
            device_map="auto"
        )

        base_model.config.pretraining_tp = 1 #parallel GPU

        model = PeftModel.from_pretrained(
            base_model,
            llm_model
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            # add_eos_token = True,
            add_bos_token = True
        )

        tokenizer.pad_token = tokenizer.eos_token

        # load model 
        model = AutoModelForCausalLM.from_pretrained(
            llm_model, 
            quantization_config=bnb_config, 
            use_cache=True, 
            device_map="auto"
        )

        model.config.pretraining_tp = 1 #parallel GPU

    return model, tokenizer



def generate_response(model, tokenizer, inputs):
    model.eval()
    input_ids = tokenizer(inputs, return_tensors='pt').input_ids.to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask = torch.where(input_ids == 2, 0, 1),
            max_new_tokens=2048,
            do_sample=False, 
            # top_p=0.9,
            # temperature=0.5
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result