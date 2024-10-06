# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd

try:
    model_name = "SamLowe/roberta-base-go_emotions"
    classifier = pipeline("text-classification",
                          model=model_name, return_all_scores=True)
    print("RoBERTa emotion model loaded successfully!")

    while True:
        user_input = input(
            "Enter a text to analyze emotions (or 'quit' to exit): ")

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        results = classifier(user_input)[0]

        # Convert results to a DataFrame for easier manipulation
        df = pd.DataFrame(results).sort_values(
            'score', ascending=False).reset_index(drop=True)

        print("\nEmotion Analysis Results:")
        print(f"Input text: '{user_input}'\n")

        # Display top 5 emotions with bar chart
        top_5 = df.head(5)
        for _, row in top_5.iterrows():
            bar_length = int(row['score'] * 50)  # Scale bar length
            print(f"{row['label']:<15} {row['score']:.4f} {'|' * bar_length}")

        print("\nAll emotions:")
        print(df.to_string(index=False))

        print("\n" + "-"*50)

except Exception as e:
    print(f"Error: {str(e)}")
