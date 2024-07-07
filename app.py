# Import necessary libraries
import gradio as gr  # Gradio is used to create a web interface to interact with the model.
import pandas as pd  # Pandas is used for data manipulation and analysis.
from datasets import load_dataset  # This function loads datasets from Hugging Face.
from sentence_transformers import SentenceTransformer  # Used for generating text embeddings.
from transformers import AutoTokenizer, AutoModelForCausalLM  # Transformers are used for natural language processing tasks.
import pymongo  # Pymongo is used to interact with MongoDB.
import os  # Used for accessing environment variables.

# Load Dataset from Hugging Face and convert it to a pandas DataFrame
# The dataset contains movie information and is split to use 80% for training.
dataset = load_dataset("MongoDB/embedded_movies", split='train[:80%]') # AIatMongoDB/embedded_movies
dataset_df = pd.DataFrame(dataset)

# Remove rows where the 'fullplot' column is empty
# It's crucial to ensure that every movie entry has a complete plot description.
# The 'fullplot' column is necessary for generating embeddings, which are numerical representations of the text.
# If this data is missing, the embeddings will be incomplete or nonsensical, reducing the accuracy of recommendations.
dataset_df = dataset_df.dropna(subset=["fullplot"])

# Drop the 'plot_embedding' column as we will generate new embeddings
# We drop the existing 'plot_embedding' column to create new embeddings using a different, potentially more effective model.
# This step ensures consistency and accuracy in the embeddings used for similarity searches.
dataset_df = dataset_df.drop(columns=["plot_embedding"])

# Load a pre-trained embedding model
# We use a pre-trained model from Sentence Transformers to convert movie plots into numerical embeddings.
# These embeddings capture the semantic content of the plots, allowing us to perform efficient and meaningful similarity searches.
embedding_model = SentenceTransformer("thenlper/gte-large")

# Define a function to generate embeddings for a given text
# Embeddings are numerical representations of text that capture its semantic meaning.
# This function checks if the text is not empty and then generates an embedding using the loaded model.
def get_embedding(text: str) -> list:
    if not text.strip():  # Check if the text is not empty
        # If the text is empty, return an empty list as it does not make sense to generate embeddings for empty text.
        # This ensures that we avoid errors and meaningless embeddings.
        print("Attempted to get embedding for empty text.")
        return []
    embedding = embedding_model.encode(text)  # Generate the embedding
    return embedding.tolist()  # Convert embedding to a list for storage and manipulation

# Apply the embedding function to the 'fullplot' column in the DataFrame
# This step generates embeddings for each movie plot in the dataset, storing them in the DataFrame for later use in similarity searches.
dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)

# Function to connect to MongoDB
# MongoDB is a NoSQL database used to store and retrieve large datasets efficiently.
# This function attempts to create a MongoDB client to connect to the database.
def get_mongo_client(mongo_uri):
    try:
        client = pymongo.MongoClient(mongo_uri)  # Create a MongoDB client
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        # Handle potential connection failures to provide feedback in case of issues with the MongoDB URI or network problems.
        print(f"Connection failed: {e}")
        return None

# Get the MongoDB URI from environment variables
# The MongoDB URI is required to connect to the database. It should be stored securely in environment variables to protect sensitive information.
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

# Connect to MongoDB using the URI
# The client connects to the 'movies' database and accesses the 'movie_collection_2' collection.
# This collection will store the movie data with their respective embeddings.
mongo_client = get_mongo_client(mongo_uri)
db = mongo_client["movies"]  # Access the 'movies' database
collection = db["movie_collection_2"]  # Access the 'movie_collection_2' collection

# Clear the collection and insert the new data
# Clearing the collection to avoid duplication of records and ensure we start with a fresh set of data.
# This step ensures that the collection only contains the most recent data with newly generated embeddings.
collection.delete_many({})  # Delete any existing records in the collection
documents = dataset_df.to_dict("records")  # Convert DataFrame to list of dictionaries
collection.insert_many(documents)  # Insert documents into the collection
print("Data ingestion into MongoDB completed")

# Function to perform a vector search on the user query
# This function generates an embedding for the user's query and uses it to search for similar movie plots in the MongoDB collection.
# Vector search allows us to find movies with plots that are semantically similar to the query.
def vector_search(user_query, collection):
    query_embedding = get_embedding(user_query)  # Generate embedding for the user query
    if query_embedding is None:
        # Return an error message if the embedding generation fails, ensuring graceful handling of invalid queries.
        return "Invalid query or embedding generation failed."

    # Define the MongoDB aggregation pipeline for vector search
    # This pipeline uses the generated query embedding to search for similar embeddings in the collection.
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # Name of the vector index
                "queryVector": query_embedding,  # Embedding of the user query
                "path": "embedding",  # Path to the embedding field in the documents
                "numCandidates": 150,  # Number of candidate matches to consider for broad retrieval
                "limit": 4,  # Return top 4 matches to keep results concise and relevant
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the '_id' field from the results for cleaner output
                "fullplot": 1,  # Include the 'fullplot' field in the results for detailed descriptions
                "title": 1,  # Include the 'title' field in the results to identify movies
                "genres": 1,  # Include the 'genres' field in the results for additional context
                "score": {"$meta": "vectorSearchScore"},  # Include the search score to assess relevance
            }
        },
    ]
    results = collection.aggregate(pipeline)  # Execute the aggregation pipeline
    return list(results)  # Return the results as a list

# Function to format search results
# This function formats the search results into a user-friendly format, making it easier for users to read and understand the recommendations.
def get_search_result(query):
    get_knowledge = vector_search(query, collection)  # Perform vector search on the query
    search_result = ""
    for result in get_knowledge:  # Iterate through search results
        # Format the search results to be user-friendly, including only the first 200 characters of the plot for brevity.
        search_result += f"Title: {result.get('title', 'N/A')}\nGenres: {', '.join(result.get('genres', ['N/A']))}\nPlot: {result.get('fullplot', 'N/A')[:200]}...\n\n"
    return search_result

# Load a pre-trained language model for generating responses
# Using GPT-2 to generate human-like responses based on the search results.
# The tokenizer converts text to a format that the model can understand, and the model generates responses.
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Function to generate a response based on the user's query
# This function combines the search results with the user's query and generates a response using the GPT-2 model.
def generate_response(query):
    source_information = get_search_result(query)  # Get search results for the query
    combined_information = (
        f"Answer the question '{query}' based on these movie details:\n\n{source_information}"
    )

    # Prepare input for the language model
    # Ensures the input does not exceed the model's maximum token capacity.
    max_length = tokenizer.model_max_length  # Get the maximum token length to ensure input does not exceed model capacity
    input_ids = tokenizer(combined_information, return_tensors="pt", max_length=max_length, truncation=True)

    try:
        response = model.generate(
            **input_ids,
            max_new_tokens=150,  # Limit the number of tokens to generate to control response length
            num_return_sequences=1,  # Generate a single response sequence
            no_repeat_ngram_size=2,  # Avoid repeating n-grams to improve response quality
            top_k=50,  # Use top-k sampling for diversity in responses
            top_p=0.95,  # Use nucleus sampling to focus on high-probability words
            temperature=0.7,  # Control the randomness of predictions to balance between creativity and coherence
            do_sample=True  # Enable sampling
        )
        return tokenizer.decode(response[0], skip_special_tokens=True)  # Decode and return the response
    except Exception as e:
        # Handle potential errors during generation and provide a meaningful error message.
        return f"An error occurred: {str(e)}"

# Function to handle user queries and generate responses
# This function ties together the query handling and response generation processes.
def query_movie_db(user_query):
    return generate_response(user_query)

# Create the Gradio interface
# Gradio provides a simple interface to interact with the model, allowing users to enter queries and receive responses.
iface = gr.Interface(
    fn=query_movie_db,  # Function to handle user queries
    inputs=gr.Textbox(lines=2, placeholder="Enter your movie query here..."),  # Textbox input for user queries
    outputs="text",  # Text output for responses
    title="Movie Recommendation Bot",  # Title of the interface
    description="Ask about movies and get detailed responses. Fore more details, visit my [blog post](https://www.kunal-pathak.com/blog/Movie-Recommendation-Bot/).",  # Description of the interface
    examples=[["Suggest me a scary movie?"], ["What action movie can I watch?"]],  # Example queries
    article="""
**My Movie Recommendation Bot** provides quick responses based on your queries but sometimes truncates replies due to token limitations in the free tier of Hugging Face resources.

This is not a coding issue but a result of operating within the token limitations of the free tier of Hugging Face resources.

To enhance response quality, better models and more resources could be used, but these come with higher costs, which I want to avoid as this is a hobby project.
    """
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()  # This launches the Gradio interface.
