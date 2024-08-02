import torch

from transformers import pipeline


# =============================================================================================================

class LLMPrompter:
    def __init__(self, llm_model, tokenizer, model):
        self.llm_model = llm_model.lower()
        self.model = model
        self.tokenizer = tokenizer

        self.initialize_chat_template()

    def initialize_chat_template(self):
        if self.llm_model.find('gpt') > -1:
            self.chat_template = [
                {"role": "system", "content": "You are a helpful assistant. Always answer the question even if the context is not helpful"},
            ]
            # self.chat_template = lambda query: [
            #     {"role": "system", "content": "You are a helpful assistant. Always answer the question even if the context is not helpful"},
            #     {"role": "user", "content": "How old is my dad?"},
            #     # {"role": "system", "content": "I think he is 50 years old."},
            #     # {"role": "user", "content": "No, my mum says he is 55 years old."},
            #     # {"role": "system", "content": "Okay, I believe your mum knows that better than I."},
            #     # {"role": "user", "content": "Now, tell me how old is my dad?"}
            # ]
        else:
            self.chat_template = [
                {"role": "user", "content": "Given that 'Beijing is the capital of China.', what is the capital of China?"},
                {"role": "assistant", "content": "Beijing"}
            ]

    def set_chat_template(self, role, content, keep_history=False):
        if not keep_history:
            self.initialize_chat_template()

        self.chat_template += [{"role": role, "content": content}]

        return self.chat_template

    def get_llm_reader(self):
        # Set model to the evaluation mode
        if hasattr(self.model, 'eval'):  # OpenAI API has no eval, just trained models
            self.model.eval()

        # Get prompt template and engine based on LLM model
        query_encoder = lambda query: query  # Default

        # This user-assisant chat instance could be more accessable
        # Mistral: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        # User-assistant chat template
        # Gemma: https://huggingface.co/google/gemma-2-9b-it
        if self.llm_model.find('mistral') > -1:
            if self.llm_model.find('-instruct') > -1:
                query_encoder = lambda query: \
                    self.tokenizer.apply_chat_template(
                        self.set_chat_template("user", query),
                        return_tensors="pt"
                    ).to('cuda')

                llm_reader = lambda query: \
                    self.tokenizer.decode(
                        self.model.generate(
                            query_encoder(query),
                            max_new_tokens=1000,
                            do_sample=False,
                            # temperature=0.2,
                            pad_token_id=self.tokenizer.pad_token_id
                        )[0]
                    )
            else:
                if self.llm_model.find('finetune') > -1:
                    input_ids = lambda query: self.tokenizer(
                        query,
                        return_tensors='pt'
                    ).input_ids.to("cuda")

                    llm_reader = lambda query: \
                        self.tokenizer.decode(
                            self.model.generate(
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
                        model=self.model,
                        tokenizer=self.tokenizer,
                        do_sample=False,
                        # temperature=0.2,  # block do_sample thus remove this
                        repetition_penalty=1.1,
                        return_full_text=False,
                        max_new_tokens=500,
                    )(query)[0]["generated_text"]
        elif self.llm_model.find('gemma') > -1:
            # https://medium.com/@coldstart_coder/
            # getting-started-with-googles-gemma-llm-using-huggingface-libraries-a0d826c552ae
            if self.llm_model.find('-it') > -1:
                query_encoder = lambda query: \
                    self.tokenizer.apply_chat_template(
                        self.set_chat_template("user", query),
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to('cuda')

                llm_reader = lambda query: \
                    self.tokenizer.decode(
                        self.model.generate(
                            query_encoder(query),
                            max_new_tokens=1000,
                            do_sample=False,
                            # temperature=0.2,
                            pad_token_id=self.tokenizer.pad_token_id
                        )[0],
                        skip_special_tokens=True
                    )
            else:
                # Refer to https://huggingface.co/docs/transformers/model_doc/gemma2
                query_encoder = lambda query: \
                    self.tokenizer(
                        query,
                        return_tensors="pt",
                        # max_length=30,
                        # truncation=True
                    ).input_ids.to('cuda')

                llm_reader = lambda query: \
                    self.tokenizer.decode(
                        self.model.generate(
                            query_encoder(query),
                            max_new_tokens=500,
                            do_sample=False,
                            # temperature=0.2,
                            # pad_token_id=tokenizer.pad_token_id
                        )[0],
                        skip_special_tokens=True
                    )
        elif self.llm_model.find('gpt') > -1:
            # This OpenAI model is actually Client
            llm_reader = lambda query: \
                self.model.chat.completions.create(
                    model=self.llm_model,
                    max_tokens=2048,
                    temperature=0.0,
                    messages=self.set_chat_template("user", query)
                ).choices[0].message.content

        return llm_reader

    def extract_answer_from_response(self, answer, prompt_example):
        try:
            # Parse the answer, corresponding to get_llm_reader()
            if self.llm_model.find('mistral') > -1:
                if self.llm_model.find('-instruct') > -1:
                    answer = answer.split('[/INST]')[-1].split('</s>')[0]
                else:
                    if self.llm_model.find('finetune') > -1:
                        answer = answer.split('###')[-1]
                    else:
                        if prompt_example:  # with an example in the prompt, can always parse by [INST]
                            answer = answer.split('[INST]')[0]
            elif self.llm_model.find('gemma') > -1:
                if self.llm_model.find('-it') > -1:
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

    def extract_chess_analysis_from_response(self, answer, prompt_example):
        try:
            # Parse the answer, corresponding to get_llm_reader()
            if self.llm_model.find('mistral') > -1:
                if self.llm_model.find('-instruct') > -1:
                    answer = answer.split('[/INST]')[-1].split('</s>')[0]
                else:
                    if self.llm_model.find('finetune') > -1:
                        answer = answer.split('\n')[-1]
                    else:
                        # very uncertain keywords
                        if prompt_example:
                            answer = answer.split('**Solution:**')[1]
                        else:
                            answer = answer.split('Answer')[1].split('Comment:')[0]
            elif self.llm_model.find('gemma') > -1:
                if self.llm_model.find('-it') > -1:
                    answer = answer.split('model\n')[1]
                else:
                    # very uncertain keywords
                    answer = answer.split('Answer:\n')[1]
        except:
            answer = answer

        answer = answer.replace('\n', '#linechange').strip()

        return answer

    def extract_chess_best_move_from_response(self, answer):
        try:
            if self.llm_model  == 'gpt-4o':
                answer = answer.split('**')[1].replace('.', '').split(' ')[-1]
            elif self.llm_model == 'gpt-3.5-turbo':
                answer = answer.split('is ')[-1].replace('.', '').replace('*', '').split(' ')[-1]
            elif self.llm_model.find('mistral') > -1:
                if self.llm_model.find('-instruct') > -1:
                    answer = answer.split('[/INST]')[1].split('** is')[-1].split('.</s>')[0].split('</s>')[0] \
                        .split('.')[-1].replace(' ', '').replace('*', '').replace(':', '')
            elif self.llm_model.find('gemma') > -1:
                if self.llm_model.find('-it') > -1:
                    answer = answer.split('model\n')[-1].split('** is')[-1].split('is **')[-1] \
                        .replace(' ', '').replace('*', '').replace('.', '')
        except:
            answer = answer.split('** is')[-1]

        answer = answer.replace('\n', '').strip()

        return answer