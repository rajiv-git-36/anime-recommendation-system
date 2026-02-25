import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the Saved Models ---
@st.cache_resource  
def load_data():
    anime_df = pickle.load(open('data/anime_data.pkl', 'rb'))
    similarity_matrix = pickle.load(open('data/similarity_matrix.pkl', 'rb'))
    svd_model = pickle.load(open('data/svd_model.pkl', 'rb'))
    return anime_df, similarity_matrix, svd_model

anime_df, similarity_matrix, svd_model = load_data()

# --- 2. The Hybrid Recommendation Engine ---
def hybrid_recommendation(user_id, anime_title):
    # Step A: Content-Based Filtering
    anime_index = anime_df[anime_df['name'] == anime_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[anime_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51] 
    
    anime_indices = [i[0] for i in sim_scores]
    
    # Step B: Collaborative Filtering (SVD)
    predictions = []
    for i in anime_indices:
        anime_id = anime_df.iloc[i]['anime_id']
        anime_name = anime_df.iloc[i]['name']
        predicted_rating = svd_model.predict(uid=user_id, iid=anime_id).est
        predictions.append((anime_name, predicted_rating))
        
    # Step C: Sorting and returning Top 5
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return [name for name, rating in predictions[:5]]

# --- 3. UI Styling ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             /* BACKGROUND IMAGE */
             background-image: url("https://images2.alphacoders.com/100/thumb-1920-1006672.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         
         /* GLASS BOX FOR THE TITLE */
         h1 {{
             background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent box */
             padding: 20px;
             border-radius: 15px;
             border: 2px solid rgba(255, 255, 255, 0.2);
             text-align: center;
             color: white !important;
             text-shadow: 2px 2px 4px #000000;
             margin-bottom: 20px;
         }}
         
         /* MAKING ALL OTHER TEXT POP */
         .stMarkdown, p, label, .stSelectbox, .stNumberInput {{
             color: white !important;
             text-shadow: 2px 2px 5px black; /* Strong black shadow for readability */
             font-weight: bold;
             font-size: 18px;
         }}
         
         /* STYLING THE CARDS  */
         .glass-card {{
             background-color: rgba(0, 0, 0, 0.7);
             padding: 15px;
             border-radius: 10px;
             border: 1px solid rgba(255, 255, 255, 0.2);
             text-align: center;
             margin-bottom: 10px;
             height: 150px;
             display: flex;
             flex-direction: column;
             justify-content: center;
             align-items: center;
             transition: transform 0.2s;
         }}
         
         .glass-card:hover {{
             transform: scale(1.05); /* Zoom effect on hover */
             border-color: #FFD700; /* Gold border */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()

# --- 4. The App Layout ---
st.title("Anime Recommendation System")

st.sidebar.header("User Options")
user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=10000, value=1)
selected_anime = st.selectbox("Select an Anime you have watched:", anime_df['name'].values)


if st.button("Recommend Anime"):
    with st.spinner("Calculating hybrid scores..."):
        recommendations = hybrid_recommendation(user_id, selected_anime)
        
        st.markdown(f"The Top 5 Picks for User {user_id}:")
        
        # Creating 5 columns for the cards
        cols = st.columns(5)
        
        for i, anime in enumerate(recommendations):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(0, 0, 0, 0.6);
                        padding: 10px;
                        border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        text-align: center;
                        height: 150px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;">
                        <h3 style="color: #FFD700; margin:0; text-shadow: 2px 2px 4px black;">#{i+1}</h3>
                        <p style="color: white; font-weight: bold; font-size: 14px; margin:0; text-shadow: 1px 1px 2px black;">{anime}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )