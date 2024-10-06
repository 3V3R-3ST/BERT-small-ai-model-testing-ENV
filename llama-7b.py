from transformers import pipeline

try:
    print("Loading Llama-7b model... This may take a while.")
    pipe = pipeline("text-generation", model="huggyllama/llama-7b")
    print("Model loaded successfully!")

    messages = []

    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = pipe(messages, max_length=100, num_return_sequences=1)

        # Extract the generated text
        generated_text = response[0]['generated_text']

        # Add the model's response to the messages
        messages.append({"role": "assistant", "content": generated_text})

        print("\nLlama-7b response:")
        print(generated_text)

        print("\n" + "-"*50)

except Exception as e:
    print(f"Error: {str(e)}")
