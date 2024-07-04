
### Recommendation System 1: SVD (Singular Value Decomposition) Model

**Description**: This model utilizes matrix factorization techniques to decompose the user-item interaction matrix and make recommendations.

**Dataset**: MovieLens 100k dataset.

**Features**:

1. **Loads the MovieLens dataset using TensorFlow Datasets**:
   - Utilizes TensorFlow Datasets to load and preprocess the MovieLens 100k dataset, which contains user-movie ratings.

2. **Prepares user IDs and movie titles vocabularies**:
   - Constructs vocabularies for user IDs and movie titles to transform categorical data into numerical format, facilitating the matrix factorization process.

3. **Defines user and movie embedding models**:
   - Creates embedding layers for users and movies to learn their latent factors, which represent the underlying preferences of users and the attributes of movies.

4. **Applies Singular Value Decomposition (SVD) for matrix factorization**:
   - Decomposes the user-item interaction matrix into three matrices: user matrix, singular values matrix, and item matrix. This helps in capturing the latent structure in the data.

5. **Reconstructs the interaction matrix and recommends movies**:
   - Reconstructs the user-item interaction matrix using the decomposed matrices to predict missing ratings. Uses these predicted ratings to generate personalized movie recommendations for users.

### Recommendation System 2: EnhancedMovieLensModel

**Description**: This model leverages a neural network architecture with additional features for more accurate movie recommendations.

**Dataset**: MovieLens 100k dataset.

**Features**:

1. **Loads the MovieLens dataset using TensorFlow Datasets**:
   - Utilizes TensorFlow Datasets to load and preprocess the MovieLens 100k dataset, which includes user-movie ratings.

2. **Prepares user IDs and movie titles vocabularies**:
   - Constructs vocabularies for user IDs and movie titles to encode categorical data into numerical format, enabling efficient model training.

3. **Defines user and movie embedding models**:
   - Creates embedding layers for users and movies to learn their latent representations, capturing the underlying preferences and attributes.

4. **Implements a retrieval task with enhanced neural network architecture**:
   - Uses a neural network model with additional layers and features to improve the representation learning and retrieval accuracy.

5. **Trains the model on the dataset and evaluates by recommending movies for sample users**:
   - Trains the model using the preprocessed dataset and evaluates its performance by generating movie recommendations for sample users, ensuring the effectiveness of the recommendations.
  
### Recommendation System 3: Genre-based Movie Recommendation Model

**Description**: This model uses genre information to recommend movies based on their genre similarity.

**Dataset**: A sample dataset containing movie titles and their associated genres.

**Features**:

1. **Loads the movie dataset using Pandas**:
   - Utilizes Pandas to load and preprocess the movie dataset, containing movie titles and genres.

2. **Prepares a TF-IDF Vectorizer for genre transformation**:
   - Uses the TfidfVectorizer from scikit-learn to transform genres into numerical vectors, facilitating the comparison of genre similarities.

3. **Computes the cosine similarity matrix**:
   - Calculates the cosine similarity matrix based on the TF-IDF vectors of the genres, providing a measure of similarity between movies.

4. **Implements a function to recommend movies based on genre similarity**:
   - Defines a function that recommends movies by finding the most similar movies based on the computed cosine similarity scores.

5. **Evaluates the model by recommending movies for a sample movie title**:
   - Demonstrates the effectiveness of the recommendation system by providing movie recommendations for a given movie title from the dataset.
