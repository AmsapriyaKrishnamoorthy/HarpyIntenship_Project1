### Recommendation System 1: SVD (Singular Value Decomposition) Model

**Description**: This model utilizes matrix factorization techniques to decompose the user-item interaction matrix and make recommendations.

**Dataset**: MovieLens 100k dataset.

**Features**:

1. **Loads and preprocesses MovieLens 100k dataset for user-movie ratings**:
   - The dataset is transformed into a user-item interaction matrix where rows represent users and columns represent movies. The matrix is filled with corresponding ratings.

2. **Applies Singular Value Decomposition (SVD)**:
   - The user-item interaction matrix is factorized into three matrices: user matrix, singular values matrix, and item matrix. This decomposition helps to capture the latent factors of users and items.

3. **Reconstructs the user-item interaction matrix**:
   - Using the decomposed matrices, the interaction matrix is reconstructed to predict missing ratings. These predicted ratings are used to make recommendations.

4. **Evaluates model performance using metrics like RMSE**:
   - The model's accuracy is assessed using metrics such as Root Mean Squared Error (RMSE) to evaluate how well the predicted ratings match the actual ratings.

5. **Recommends movies to users based on predicted ratings**:
   - Generates personalized movie recommendations for users by predicting ratings for unrated movies and recommending the top-rated ones.

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
  
### Genre-based Movie Recommendation Model

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
