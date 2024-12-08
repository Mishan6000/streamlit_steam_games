import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import requests

def main():
  data = pd.read_csv('steam.csv')
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)

  st.title("Research on the Statistics of the Gaming Industry on Steam")

  st.write(data.head(5))

  st.write("""**Data description:**
  
  *  'appid' - Unique identifier for each title
  
  *  'name' - Title of app (game)
  
  *  'release_date' - Release date in format YYYY-MM-DD
  
  *  'english' - Language support: 1 if is in English
  
  *  'developer' - Name (or names) of developer(s). Semicolon delimited if multiple
  
  *  'publisher' - Name (or names) of publisher(s). Semicolon delimited if multiple
  
  *  'platforms' - Semicolon delimited list of supported platforms. At most includes: windows;mac;linux
  
  *  'required_age' - Minimum required age according to PEGI UK standards. Many with 0 are unrated or unsupplied.
  
  *  'categories' - Semicolon delimited list of game categories, e.g. single-player;multi-player
  
  *  'genres' - Semicolon delimited list of game genres, e.g. action;adventure
  
  *  'steamspy_tags' - Semicolon delimited list of top steamspy game tags, similar to genres but community voted, e.g. action;adventure
  
  *  'achievements' - Number of in-game achievements, if any
  
  *  'positive_ratings' - Number of positive ratings, from SteamSpy
  
  *  'negative_ratings' - Number of negative ratings, from SteamSpy
  
  *  'average_playtime' - Average user playtime, from SteamSpy
  
  *  'median_playtime' - Median user playtime, from SteamSpy
  
  *  'owners' - Estimated number of owners. Contains lower and upper bound (like 20000-50000). May wish to take mid-point or lower bound. Included both to give options.
  
  *  'price' - Current full price of title in GBP, (pounds sterling)
  
  
  """)

  url = "http://127.0.0.1:8000"

  st.header("Appendix: FastAPI - Game Data Management")

  # Section for fetching game data
  st.markdown("#### Get Sample from the Dataset")
  start = st.number_input("Start Index", min_value=0, value=0)
  limit = st.number_input("Number of Rows to Fetch", min_value=1, value=10)
  developer_filter = st.text_input("Filter by Developer")

  if st.button("Fetch Data"):
    params = {"start": start, "limit": limit}
    if developer_filter:
      params["developer"] = developer_filter
    response = requests.get(f"{url}/games/", params=params)

    if response.status_code == 200:
      df = response.json()
      st.write(pd.DataFrame(df))
    else:
      st.error("!!! ERROR !!!")

  # Section for appending a new game entry
  st.markdown("#### Append New Game Entry")
  new_game_entry = {
    "name": st.text_input("Game Name"),
    "release_date": st.text_input("Release Date (YYYY-MM-DD)"),
    "english": st.number_input("English Support (1 for Yes, 0 for No)", min_value=0, max_value=1, value=1),
    "developer": st.text_input("Developer"),
    "publisher": st.text_input("Publisher"),
    "platforms": st.text_input("Platforms (comma-separated)"),
    "required_age": st.number_input("Required Age", min_value=0),
    "categories": st.text_input("Categories (comma-separated)"),
    "genres": st.text_input("Genres (comma-separated)"),
    "steamspy_tags": st.text_input("SteamSpy Tags (comma-separated)"),
    "achievements": st.number_input("Achievements", min_value=0),
    "positive_ratings": st.number_input("Positive Ratings", min_value=0),
    "negative_ratings": st.number_input("Negative Ratings", min_value=0),
    "average_playtime": st.number_input("Average Playtime (in minutes)", min_value=0),
    "median_playtime": st.number_input("Median Playtime (in minutes)", min_value=0),
    "owners": st.text_input("Owners (e.g., '1M-2M')"),
    "price": st.number_input("Price", min_value=0.0, format="%.2f")
  }

  if st.button("Submit New Game"):
    response = requests.post(f"{url}/games/", json=new_game_entry)

    if response.status_code == 200:
      st.success("Successfully Added!")
    else:
      st.error("!!! ERROR !!!")

  st.write(data.describe())

  st.write(data.isna().any())
  st.write("No NaN values in the dataset")

  st.write(data.dtypes)

  def preproc(df, cols_to_le, cols_to_ohe):

    df['release_date'] = pd.to_datetime(df['release_date']).astype(int) / 10**9
    df['owners'] = df['owners'].apply(lambda x: int((np.mean(list(map(int, x.split('-')))))))

    for col in cols_to_ohe:
      df[col] = df[col].apply(lambda x: x.split(';'))
      df = df.explode(col)
      df = pd.concat([df, pd.get_dummies(df[col], prefix=col[:-1], prefix_sep='_', dtype='int')], axis=1).groupby('appid').max().reset_index()
      df = df.drop(columns=[col])

    le = LabelEncoder()

    for col in cols_to_le:
      df[col] = le.fit_transform(df[col])

    df['rating'] = df['positive_ratings'] - df['negative_ratings'] #Let's create a rating feature

    df = df.drop(columns=['name'])

    return df

  cols_to_le = ['developer', 'publisher']
  cols_to_ohe = ['platforms', 'categories', 'genres', 'steamspy_tags']

  train = preproc(data, cols_to_le, cols_to_ohe)

  train_data = train[['appid', 'release_date', 'english', 'developer', 'publisher',
                      'required_age', 'achievements', 'positive_ratings', 'negative_ratings',
                      'average_playtime', 'median_playtime', 'owners', 'price', 'rating']]

  show_dist = ['rating', 'price', 'owners']

  for option in show_dist:
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    plt.title(f'Распределение переменной <{option}>')
    sns.histplot(x=train[option], bins=len(np.unique(train[option])), kde=True, stat='density', ax=ax_hist)
    sns.boxplot(x=train[option], ax=ax_box)

    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_box.set(xlabel='')
    plt.ylabel('Плотность')
    plt.xscale('log')
    st.pyplot(f)

  st.write("""Hypothesis: the success of a game, which can be assessed by its rating and number of downloads, is determined by genre characteristics, various tags, and other factors.""")

  df = pd.DataFrame({"Rating" : train["rating"], "Owners" : train["owners"], "Price" : train["price"], "Realese_date" : train["release_date"]})
  fig = px.scatter_3d(df, x='Realese_date', y='Rating', z='Price', color='Owners', log_y=True, color_discrete_sequence=px.colors.qualitative.Alphabet)
  fig

  f, ax = plt.subplots()
  plt.title('Owners correlation')
  sns.scatterplot(x='rating', y='owners', data=train, ax=ax)
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Rating', fontsize=10)
  plt.xlabel('Owners', fontsize=10)
  st.pyplot(f)

  f, ax = plt.subplots(figsize=(12, 12))
  plt.title('Correlation')
  sns.heatmap(train_data.corr(), cmap="crest", annot=True, fmt=".1f", ax=ax)
  st.pyplot(f)

  corr_matrix = train.corr(method='pearson')

  f, ax = plt.subplots()
  plt.title('Owners correlation')
  corr_matrix[(corr_matrix.abs() > 0.1) & (corr_matrix != 1)].unstack().sort_values()['owners'].dropna().plot(kind='barh')
  st.pyplot(f)

  f, ax = plt.subplots()
  plt.title('Rating correlation')
  corr_matrix[(corr_matrix.abs() > 0.1) & (corr_matrix != 1)].unstack().sort_values()['rating'].dropna().plot(kind='barh')
  st.pyplot(f)

if __name__ == "__main__":
    main()
