from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import html

app = Flask(__name__)
CORS(app)

# --- MODEL AND DATA LOADING ---
try:
    with open('dt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    print("✅ Model columns loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model columns: {e}")
    model_columns = None

try:
    df = pd.read_csv('anime.csv')
    df['name'] = df['name'].apply(html.unescape)
    df.dropna(subset=['rating', 'genre', 'members'], inplace=True)
    df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0])
    print(f"✅ Dataset loaded and cleaned: {len(df)} animes")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None

# -------------------------------------------------------------------
# --- START: UPDATED FILTERS SECTION ---
# -------------------------------------------------------------------
@app.route('/api/filters', methods=['GET'])
def get_filters():
    """Get the specific lists of genres, types, and single ratings."""
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    try:
        genre_list = [
            'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror',
            'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports',
            'Supernatural', 'Thriller'
        ]
        type_list = ['TV', 'Movie', 'OVA', 'Special']
        
        # Use a list of single numbers for the minimum rating
        ratings = [9, 8, 7, 6, 5, 4, 3, 2, 1]

        return jsonify({
            'genres': genre_list,
            'types': type_list,
            'ratings': ratings
        }), 200

    except Exception as e:
        print(f"Error in /api/filters: {e}")
        return jsonify({'error': str(e)}), 500
# -------------------------------------------------------------------
# --- END: UPDATED FILTERS SECTION ---
# -------------------------------------------------------------------


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get anime recommendations based on user preferences"""
    if df is None or model is None or model_columns is None:
        return jsonify({'error': 'Server is not ready: model or data not loaded'}), 500

    try:
        data = request.json
        genre = data.get('genre', 'All')
        anime_type = data.get('type', 'All')
        # Get the single rating value from the frontend
        rating_value = data.get('rating')

        filtered_df = df.copy()

        if genre and genre != 'All':
            filtered_df = filtered_df[filtered_df['genre'].str.contains(genre, case=False, na=False)]

        if anime_type and anime_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == anime_type]

        # Filter by minimum rating if a value is provided by the user
        if rating_value:
            min_rating = float(rating_value)
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        if filtered_df.empty:
            return jsonify({'recommendations': [], 'count': 0}), 200

        X_to_predict = filtered_df[['primary_genre', 'members']]
        X_to_predict_encoded = pd.get_dummies(X_to_predict, columns=['primary_genre'])
        X_to_predict_aligned = X_to_predict_encoded.reindex(columns=model_columns, fill_value=0)

        predictions = model.predict(X_to_predict_aligned)
        filtered_df['prediction'] = predictions

        recommendations = filtered_df[filtered_df['prediction'] == 1]
        recommendations = recommendations.sort_values(by='rating', ascending=False)
        final_recommendations = recommendations.head(20)

        result = []
        for _, row in final_recommendations.iterrows():
            result.append({
                'name': row.get('name', 'N/A'),
                'genre': row.get('genre', 'N/A'),
                'type': row.get('type', 'N/A'),
                'rating': float(row.get('rating', 0)),
            })
            
        return jsonify({
            'recommendations': result,
            'count': len(result)
        }), 200
        
    except Exception as e:
        print(f"Error in /api/recommend: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
