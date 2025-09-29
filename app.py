from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained Decision Tree model
try:
    with open('dt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load the anime dataset
try:
    df = pd.read_csv('anime.csv')
    print(f"✅ Dataset loaded: {len(df)} animes")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None

# Get unique values for dropdowns
def get_unique_values():
    if df is None:
        return {
            'genres': [],
            'types': [],
            'ratings': []
        }
    
    # Extract unique genres (assuming genres are comma-separated)
    all_genres = set()
    if 'genre' in df.columns:
        for genres in df['genre'].dropna():
            if isinstance(genres, str):
                all_genres.update([g.strip() for g in genres.split(',')])
    
    # Get unique types
    types = df['type'].dropna().unique().tolist() if 'type' in df.columns else []
    
    # Create rating ranges
    ratings = ['All', '1-3', '3-5', '5-7', '7-9', '9-10']
    
    return {
        'genres': sorted(list(all_genres)),
        'types': sorted(types),
        'ratings': ratings
    }

@app.route('/api/filters', methods=['GET'])
def get_filters():
    """Get available filter options"""
    try:
        filters = get_unique_values()
        return jsonify(filters), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get anime recommendations based on user preferences"""
    try:
        data = request.json
        genre = data.get('genre')
        anime_type = data.get('type')
        rating_range = data.get('rating')
        
        if df is None:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Filter the dataset based on user input
        filtered_df = df.copy()
        
        # Filter by genre
        if genre and genre != 'All':
            filtered_df = filtered_df[filtered_df['genre'].str.contains(genre, case=False, na=False)]
        
        # Filter by type
        if anime_type and anime_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == anime_type]
        
        # Filter by rating range
        if rating_range and rating_range != 'All':
            if 'rating' in filtered_df.columns:
                if rating_range == '1-3':
                    filtered_df = filtered_df[(filtered_df['rating'] >= 1) & (filtered_df['rating'] < 3)]
                elif rating_range == '3-5':
                    filtered_df = filtered_df[(filtered_df['rating'] >= 3) & (filtered_df['rating'] < 5)]
                elif rating_range == '5-7':
                    filtered_df = filtered_df[(filtered_df['rating'] >= 5) & (filtered_df['rating'] < 7)]
                elif rating_range == '7-9':
                    filtered_df = filtered_df[(filtered_df['rating'] >= 7) & (filtered_df['rating'] < 9)]
                elif rating_range == '9-10':
                    filtered_df = filtered_df[(filtered_df['rating'] >= 9) & (filtered_df['rating'] <= 10)]
        
        # If model exists and we have features, use it for ranking
        if model is not None and len(filtered_df) > 0:
            # Prepare features for prediction (adjust based on your model's features)
            # This is a placeholder - adjust according to your actual model features
            try:
                # Sort by rating if available, otherwise random
                if 'rating' in filtered_df.columns:
                    filtered_df = filtered_df.sort_values('rating', ascending=False)
                else:
                    filtered_df = filtered_df.sample(frac=1)
            except:
                pass
        
        # Get top recommendations (limit to 20)
        recommendations = filtered_df.head(20)
        
        # Prepare response
        result = []
        for _, row in recommendations.iterrows():
            anime_info = {
                'name': row.get('name', 'Unknown'),
                'genre': row.get('genre', 'Unknown'),
                'type': row.get('type', 'Unknown'),
                'episodes': int(row.get('episodes', 0)) if pd.notna(row.get('episodes')) else 0,
                'rating': float(row.get('rating', 0)) if pd.notna(row.get('rating')) else 0,
                'members': int(row.get('members', 0)) if pd.notna(row.get('members')) else 0
            }
            result.append(anime_info)
        
        return jsonify({
            'recommendations': result,
            'count': len(result)
        }), 200
        
    except Exception as e:
        print(f"Error in recommend: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': df is not None,
        'dataset_size': len(df) if df is not None else 0
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)