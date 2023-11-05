def infer_lm():
    model_name = "model_ft/fine_tuned_llama-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Enter a prompt to generate text or type 'exit' to quit.")
    while True:
        prompt = input("Prompt: ")

        if prompt.lower() == "exit":
            break

        input_tokens = tokenizer.encode(prompt, return_tensors="pt")
        output_tokens = model.generate(input_tokens, max_length=20, num_return_sequences=1)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        print(f"Generated text: {output_text}")

infer_lm()