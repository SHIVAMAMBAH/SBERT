import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load FAQs from JSON file
def load_faqs(json_file):
    with open(json_file, 'r') as file:
        faqs = json.load(file)
    return faqs

# Function to get the most similar FAQ answer
def get_answer(query, faqs):
    # Vectorize the input query and FAQ questions
    query_embedding = model.encode([query])
    faq_questions = [faq['question'] for faq in faqs]
    faq_embeddings = model.encode(faq_questions)

    # Calculate cosine similarity between the query and each FAQ question
    similarities = cosine_similarity(query_embedding, faq_embeddings)

    # Find the FAQ with the highest similarity
    best_match_idx = np.argmax(similarities)

    # Return the answer corresponding to the most similar FAQ
    return faqs[best_match_idx]['answer']

# Main execution function
def main():
    # Specify your JSON file path
    json_file = 'faq_data.json'  # Replace with your JSON file path

    # Load FAQs
    faqs = load_faqs(json_file)

    # Take user input for the query
    query = input("Please enter your question: ")

    # Get the best matching answer
    answer = get_answer(query, faqs)

    # Output the answer
    print("Answer: ", answer)

if __name__ == "__main__":
    main()
