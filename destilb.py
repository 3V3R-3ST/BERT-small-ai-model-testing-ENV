from transformers import pipeline, AutoTokenizer

try:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    pipe = pipeline("text-classification", model=model_name,
                    return_all_scores=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")

    while True:
        # Get user input
        user_input = input(
            "Enter a sentence to classify (or 'quit' to exit): ")

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        # Tokenize the input
        tokens = tokenizer.encode(user_input, add_special_tokens=True)
        token_count = len(tokens)

        # Classify the input
        results = pipe(user_input)

        # Print the results
        print("\nClassification Results:")
        print(f"Input text: '{user_input}'")
        print(f"Token count: {token_count}")

        print("\nAll scores:")
        for result in results[0]:
            print(f"- {result['label']}: {result['score']:.4f}")

        # Print the highest scoring classification
        top_result = max(results[0], key=lambda x: x['score'])
        print(f"\nTop classification: {top_result['label']}")
        print(f"Confidence: {top_result['score']:.4f}")

        print("\n" + "-"*50)  # Separator for readability

except Exception as e:
    print(f"Error: {str(e)}")
