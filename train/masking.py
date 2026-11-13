from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant"

collator = DataCollatorForCompletionOnlyLM(response_template, )