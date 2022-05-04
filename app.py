# DASH
import dash
from dash import dcc
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
port = int(os.environ.get("PORT", 6005))

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

with open('./data/poster_urls_lists', 'rb') as fp:
  poster_lists = pickle.load(fp)

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
pairwise_similarities = cosine_similarity(document_embeddings)

##############################################
# APP LAYOUT
##############################################

app.layout = html.Div([
          html.H1('Movie Recommendation'),
          
          html.Div(children=[dcc.Dropdown(id="movie-lists",
                                          options=movie_lists,
                                          multi=False)]
                   ),
          html.Div(children=[html.Div(children=[html.H3(id='focused-movie',
                                                        style={'margin': 'auto'}),
                                                html.Div(id='focused-poster')],
                                      style={'textAlign': 'center', 'flexBasis': '100%'}),
                             html.Div(id='similar-movies',
                                      style={'textAlign': 'center', 'flexBasis': '100%', 'display': 'flex', 'width': '100%', 'justifyContent': 'space-around'})
                   ],
                   style={'display': 'flex', 'width': '90%', 'justifyContent': 'space-around'}
                   ),
          

          html.H2("We will recommend movies that are close to the story you describe."),
          html.Div(dcc.Input(id='overview', type='text',
                             placeholder='Describe the movie you want to see... (keyword, overview, story)',
                             value='Ice magic princess who lives with sister',
                             style={'width': '80%'})
          ),
          html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
          html.Div(id="movie",
                   children=[html.Div(children=[html.H3(id='moviename1', style={'marginBottom': 0}),
                                                html.H4(id='moviescore1', style={'margin': 'auto'}),
                                                html.Div(id='poster1')],
                                      style={'textAlign': 'center', 'flexBasis': '100%'}),
                             html.Div(children=[html.H3(id='moviename2', style={'marginBottom': 0}),
                                                html.H4(id='moviescore2', style={'margin': 'auto'}),
                                                html.Div(id='poster2')],
                                      style={'textAlign': 'center', 'flexBasis': '100%'}),
                             html.Div(children=[html.H3(id='moviename3', style={'marginBottom': 0}),
                                                html.H4(id='moviescore3', style={'margin': 'auto'}),
                                                html.Div(id='poster3')],
                                      style={'textAlign': 'center', 'flexBasis': '100%'})],
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
               Output('moviescore3', 'children'),
               Output('poster1', 'children'),
               Output('poster2', 'children'),
               Output('poster3', 'children')],
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
  recommended_posters = list(poster_lists[similar_idx[:topn]])
              
  scores = [' (' + str(round(similarity_scores[i]*100, 2)) + '%)' for i in range(3)]

  poster_divs = [html.Img(style={'maxHeight': '400px'}, src=url) for url in recommended_posters]

  return recommended_movies + scores + poster_divs


@app.callback([Output('focused-movie', 'children'),
               Output('focused-poster', 'children'),
               Output('similar-movies', 'children')],
              Input('movie-lists', 'value'))
def similarity(focus_movie):
  topn = 3

  focus_movie_idx = list(movie_lists).index(focus_movie)
  focus_poster_url = poster_lists[focus_movie_idx]
  focus_poster = html.Img(style={'maxHeight': '400px'}, src=focus_poster_url)

  pairwise = pairwise_similarities[focus_movie_idx]  
  similar_idxs = np.argsort(pairwise)[::-1]
  similar_scores = list(pairwise[similar_idxs[1:topn+1]])
  similar_movies = list(movie_lists[similar_idxs[1:topn+1]])
  similar_posters = list(poster_lists[similar_idxs[1:topn+1]])

  similar_div_list = [html.Div(children=[html.H3(similar_movies[i],
                                                 style={'marginBottom': 0}),
                                         html.H4('(' + str(round(similar_scores[i]*100, 2)) + '%)',
                                                 style={'margin': 'auto'}),
                                         html.Div(html.Img(style={'maxHeight': '250px'},
                                                      src=similar_posters[i]))
                                         ],
                               style={'textAlign': 'center', 'flexBasis': '100%', 'margin': '10px'}) for i in range(topn)]

  similar_div = html.Div(children=[html.H5("Similar movies"),
                                   html.Div(children=[similar_div_list[0],
                                                      similar_div_list[1],
                                                      similar_div_list[2]],
                                            style={'display': 'flex', 'width': '90%', 'justifyContent': 'space-around'}),
                                   ]
                         )

  return focus_movie, focus_poster, similar_div


if __name__ == "__main__":
    app.run_server(debug=False,
                   host="0.0.0.0",
                   port=port)