messages_helpful = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data_negative['constitution'][constitution]}\n\n{data_negative['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
        prompt_helpful = encoded_helpful.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_helpful}{data_negative['response_helpful'][response_helpful]}"
        logprob_pos_helpful = model.batch_log_probs([prompt_helpful], [response_formatted_helpful])
        
        messages_helpful_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_positive}\n\n{data_negative['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful_negative))
        prompt_negative_helpful = encoded_helpful_negative.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_negative_helpful}{data_negative['response_helpful'][response_helpful]}"
        logprob_neg_helpful = model.batch_log_probs([prompt_negative_helpful], [response_formatted_helpful])        
        
        log_probs_positive_helpful.append(logprob_pos_helpful.item())
        log_probs_negative_helpful.append(logprob_neg_helpful.item())

        
        # Harmless
        messages_harmless = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data_negative['constitution'][constitution]}\n\n{data_negative['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))
        prompt_harmless = encoded_harmless.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_harmless}{data_negative['response_harmless'][response_harmless]}"
        logprob_pos_harmless = model.batch_log_probs([prompt_harmless], [response_formatted_harmless])
        
        messages_harmless_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_positive}\n\n{data_negative['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless_negative))
        prompt_negative_harmless = encoded_harmless_negative.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_negative_harmless}{data_negative['response_harmless'][response_harmless]}"
        logprob_neg_harmless = model.batch_log_probs([prompt_negative_harmless], [response_formatted_harmless])   