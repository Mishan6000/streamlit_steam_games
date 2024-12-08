from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


app = FastAPI()

# Define the data model using Pydantic
class Game(BaseModel):
    appid: int
    name: str
    release_date: str  # You can use a more specific type like datetime.date if needed
    english: bool
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
    average_playtime: float
    median_playtime: float
    owners: str  # You might want to specify the format of this field (e.g., "1M-2M")
    price: float


# Initialize an empty DataFrame or load your existing dataset
data = pd.read_csv('steam.csv')
print(data.head(5))

@app.post("/games/")
async def create_game(game: Game):
    global data  # Use the global data variable

    # Convert the game data to a dictionary and append it to the DataFrame
    game_dict = game.dict()

    # Append the new game to the DataFrame
    data = data.append(game_dict, ignore_index=True)

    return {"message": "Game added successfully", "game": game_dict}

@app.get("/games/")
async def get_games():
    global data  # Use the global data variable

    # Convert the DataFrame to a list of dictionaries for JSON response
    games_list = data.to_dict(orient='records')

    return {"games": games_list}