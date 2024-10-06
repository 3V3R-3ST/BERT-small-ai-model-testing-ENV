# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoConfig

try:
    model_name = "abullard1/albert-v2-steam-review-constructiveness-classifier"
    config = AutoConfig.from_pretrained(model_name)

    pipe = pipeline("text-classification", model=model_name,
                    return_all_scores=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("ALBERT model and tokenizer loaded successfully!")

    # Define custom labels
    custom_labels = {
        "LABEL_0": "Not Constructive (Not Helpful/Not Meaningful)",
        "LABEL_1": "Constructive (Helpful/Meaningful)"
    }

    # Print label information
    if hasattr(config, 'id2label'):
        print("\nLabel meanings:")
        for id, label in config.id2label.items():
            print(f"- {id}: {label}")
    else:
        print("\nWarning: Label meanings not found in model config.")

    while True:
        # Get user input
        user_input = input(
            "Enter a Steam review to classify (or 'quit' to exit): ")

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
            custom_label = custom_labels[result['label']]
            print(f"- {custom_label}: {result['score']:.4f}")

        # Print the highest scoring classification
        top_result = max(results[0], key=lambda x: x['score'])
        top_custom_label = custom_labels[top_result['label']]
        print(f"\nTop classification: {top_custom_label}")
        print(f"Confidence: {top_result['score']:.4f}")

        print("\n" + "-"*50)  # Separator for readability

except Exception as e:
    print(f"Error: {str(e)}")
