from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd


app = FastAPI()

# Define the data model using Pydantic
class Game(BaseModel):
    name: str
    release_date: str  # You can use a more specific type like datetime.date if needed
    english: int
    developer: str
    publisher: str
    platforms: str  # You can use List[str] if you want to store multiple platforms
    required_age: int
    categories: str  # Again, List[str] can be used for multiple categories
    genres: str  # Same as above
    steamspy_tags: str  # Same as above
    achievements: int
    positive_ratings: int
    negative_ratings: int
    average_playtime: int
    median_playtime: int
    owners: str  # You might want to specify the format of this field (e.g., "1M-2M")
    price: float


# Initialize an empty DataFrame or load your existing dataset
data = pd.read_csv('steam.csv', sep=';')


@app.post("/games/")
def add_game(new_game: Game):
    """
    POST method to add a new game to the dataset.
    """
    global data
    # Create a new row based on the new game entry
    new_row = new_game.dict()
    new_row['appid'] = data['appid'].max() + 1  # Assign a new game_id
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    return {"message": "New game added successfully!", "data": new_row}

@app.get("/games/")
def get_games(start: int = 0, limit: int = 10, developer: str = Query(None)):
    """
    GET method with pagination and filtering by 'developer'.
    """
    filtered_df = data if developer is None else data[data['developer'] == developer]
    return filtered_df.iloc[start:start + limit].to_dict(orient="records")
