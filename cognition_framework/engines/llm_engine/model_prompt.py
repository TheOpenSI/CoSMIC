import torch

from transformers import pipeline
# from .load_model_test import generate_response


# =============================================================================================================

DEBUG = False

# =============================================================================================================

def get_llm_reader(llm_model, tokenizer, model, has_context=False):
    # Set model to the evaluation mode
    if hasattr(model, 'eval'):  # OpenAI API has no eval, just trained models
        model.eval()

    # Get LLM model
    llm_model = llm_model.lower()

    # Get prompt template and engine based on LLM model
    query_encoder = lambda query: query  # Default

    # Set chat template without and with context
    chat_template = lambda query: [
        {"role": "user", "content": query}
    ]

    if llm_model.find('gpt') > -1:
        chat_template_context = lambda query: [
            {"role": "system", "content": "You are a helpful assistant. Always answer the question even if the context is not helpful"},
            {"role": "user", "content": query}
        ]
    else:
        chat_template_context = lambda query: [
            {"role": "user", "content": "Given that 'Beijing is the capital of China.', what is the capital of China?"},
            {"role": "assistant", "content": "Beijing"},
            {"role": "user", "content": query}
        ]

    # This user-assisant chat instance could be more accessable
    # Mistral: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    # User-assistant chat template
    # Gemma: https://huggingface.co/google/gemma-2-9b-it
    if llm_model.find('mistral') > -1:
        if llm_model.find('-instruct') > -1:
            query_encoder = lambda query: \
                tokenizer.apply_chat_template(
                    chat_template_context(query) if has_context else chat_template(query),
                    return_tensors="pt"
                ).to('cuda')

            llm_reader = lambda query: \
                tokenizer.decode(
                    model.generate(
                        query_encoder(query),
                        max_new_tokens=1000,
                        do_sample=False,
                        # temperature=0.2,
                        pad_token_id=tokenizer.pad_token_id
                    )[0]
                )
        else:
            if llm_model.find('finetune') > -1:
                input_ids = lambda query: tokenizer(
                    query,
                    return_tensors='pt'
                ).input_ids.to("cuda")

                llm_reader = lambda query: \
                    tokenizer.decode(
                        model.generate(
                            input_ids=input_ids(query),
                            attention_mask=torch.where(input_ids(query) == 2, 0, 1),
                            max_new_tokens=2048,
                            do_sample=False,
                            # top_p=0.9,
                            # temperature=0.5
                        )[0],
                        skip_special_tokens=True
                    )
            else:
                llm_reader = lambda query: pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                    # temperature=0.2,  # block do_sample thus remove this
                    repetition_penalty=1.1,
                    return_full_text=False,
                    max_new_tokens=500,
                )(query)[0]["generated_text"]
    elif llm_model.find('gemma') > -1:
        # https://medium.com/@coldstart_coder/
        # getting-started-with-googles-gemma-llm-using-huggingface-libraries-a0d826c552ae
        if llm_model.find('-it') > -1:
            query_encoder = lambda query: \
                tokenizer.apply_chat_template(
                    chat_template_context(query) if has_context else chat_template(query),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to('cuda')
        
            llm_reader = lambda query: \
                tokenizer.decode(
                    model.generate(
                        query_encoder(query),
                        max_new_tokens=1000,
                        do_sample=False,
                        # temperature=0.2,
                        pad_token_id=tokenizer.pad_token_id
                    )[0],
                    skip_special_tokens=True
                )
        else:
            # Refer to https://huggingface.co/docs/transformers/model_doc/gemma2
            query_encoder = lambda query: \
                tokenizer(
                    query,
                    return_tensors="pt",
                    # max_length=30,
                    # truncation=True
                ).input_ids.to('cuda')
            
            llm_reader = lambda query: \
                tokenizer.decode(
                    model.generate(
                        query_encoder(query),
                        max_new_tokens=500,
                        do_sample=False,
                        # temperature=0.2,
                        # pad_token_id=tokenizer.pad_token_id
                    )[0],
                    skip_special_tokens=True
                )
    elif llm_model.find('gpt') > -1:
        # This OpenAI model is actually Client
        llm_reader = lambda query: \
            model.chat.completions.create(
                model=llm_model,
                max_tokens=2048,
                temperature=0.0,
                messages=chat_template_context(query)
            ).choices[0].message.content

    return llm_reader

# =============================================================================================================

def extract_answer_from_response(llm_model, answer, prompt_example):
    # Debug information
    if DEBUG: print('####', answer)

    try:  # TODO
        # Parse the answer, corresponding to get_llm_reader()
        if llm_model.find('mistral') > -1:
            if llm_model.find('-instruct') > -1:
                answer = answer.split('[/INST]')[-1].split('</s>')[0]
            else:
                if llm_model.find('finetune') > -1:
                    answer = answer.split('###')[-1]
                else:
                    if prompt_example:  # with an example in the prompt, can always parse by [INST]
                        answer = answer.split('[INST]')[0]
        elif llm_model.find('gemma') > -1:
            if llm_model.find('-it') > -1:
                answer = answer.split('model\n')[1]
            else:
                if prompt_example:
                    answer = answer.split('model\n')[2]
                else:
                    answer = answer.split('### ANSWER:\n')[-1]
    except:
        answer = answer

    answer = answer.replace('\n', '#linechange').strip()

    return answer

# =============================================================================================================

def extract_chess_answer_from_response(llm_model, answer, prompt_example):
    # Debug information
    if DEBUG: print('#### Chess', answer)

    try:  # TODO
        # Parse the answer, corresponding to get_llm_reader()
        if llm_model.find('mistral') > -1:
            if llm_model.find('-instruct') > -1:
                answer = answer.split('[/INST]')[-1].split('</s>')[0]
            else:
                if llm_model.find('finetune') > -1:
                    answer = answer.split('\n')[-1]
                else:
                    # very uncertain keywords
                    if prompt_example:
                        answer = answer.split('**Solution:**')[1]
                    else:
                        answer = answer.split('Answer')[1].split('Comment:')[0]
        elif llm_model.find('gemma') > -1:
            if llm_model.find('-it') > -1:
                answer = answer.split('model\n')[1]
            else:
                # very uncertain keywords
                answer = answer.split('Answer:\n')[1]
    except:
        answer = answer

    answer = answer.replace('\n', '#linechange').strip()

    return answer