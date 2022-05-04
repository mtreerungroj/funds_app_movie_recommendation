# DASH
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# OTHER LIBRARIES
import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
port = int(os.environ.get("PORT", 6004))

##############################################
# FUNCTIONS
##############################################

def prepare_text(x):
  return " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l)


##############################################
# DATA
##############################################

with open('./data/movie_lists', 'rb') as fp:
  movie_lists = pickle.load(fp)

with open('./data/document_embeddings', 'rb') as fp:
  document_embeddings = pickle.load(fp)

with open('./data/stopwords', 'rb') as fp:
  stop_words_l = pickle.load(fp)

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


##############################################
# APP LAYOUT
##############################################

app.layout = html.Div([
          html.H1('Movie recommendation'),
          html.H2('どんな映画が見てみたいですか？'),
          html.Div(dcc.Input(id='overview', type='text',
                             placeholder='Type an overview...',
                             style={'width': '80%'})),
          html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
          html.Div(children=[html.Div(children=[html.H3(id='moviename1', style={'marginBottom': 0}),
                                                html.H4(id='moviescore1', style={'margin': 'auto'})],
                                      style={'textAlign': 'center'}),
                             html.Div(children=[html.H3(id='moviename2', style={'marginBottom': 0}),
                                                html.H4(id='moviescore2', style={'margin': 'auto'})],
                                      style={'textAlign': 'center'}),
                             html.Div(children=[html.H3(id='moviename3', style={'marginBottom': 0}),
                                                html.H4(id='moviescore3', style={'margin': 'auto'})],
                                      style={'textAlign': 'center'})],
                   style={'display': 'flex', 'width': '90%', 'justifyContent': 'space-around'}
          )
])

##############################################
# APP CALLBACKS
##############################################

@app.callback([Output('moviename1', 'children'),
               Output('moviename2', 'children'),
               Output('moviename3', 'children'),
               Output('moviescore1', 'children'),
               Output('moviescore2', 'children'),
               Output('moviescore3', 'children')],
              Input('submit-button-state', 'n_clicks'),
              State('overview', 'value'),
              prevent_initial_call=True)
def recommendation(_, text):
  topn = 3
  prepared_text = prepare_text(text)
  embedded_text = [list(sbert_model.encode(prepared_text))]

  similarity = cosine_similarity(embedded_text, document_embeddings)[0]
  similar_idx = np.argsort(similarity)[::-1]

  similarity_scores = list(similarity[similar_idx[:topn]])
  recommended_movies = list(movie_lists[similar_idx[:topn]])
  # result = [recommended_movies[i] + ' (' + str(round(similarity_scores[i]*100, 2)) + '%)' for i in range(3)]
  scores = [' (' + str(round(similarity_scores[i]*100, 2)) + '%)' for i in range(3)]
  
  return recommended_movies+scores


if __name__ == "__main__":
    app.run_server(debug=False,
                   host="0.0.0.0",
                   port=port)