# Readability Prediction API

This project provides a RESTful API for predicting the readability of text excerpts using a machine learning model. It is built with FastAPI and utilizes a RandomForestRegressor model trained on a dataset to predict readability scores.

## Installation

To set up the project environment, follow these steps:

1. Ensure you have Python 3.8 or newer installed.
2. Clone this repository to your local machine.
3. Navigate to the project directory and install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Training the Model

Before using the API, you need to train the model with your dataset. The training script `prepare-model.py` is provided for this purpose. It processes the data, trains the model, and saves it along with the vectorizer for later use by the API.

To train the model, run:

```bash
python prepare-model.py
```

## Running the API

After training the model, you can start the FastAPI server to begin serving predictions. To start the server, run:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Endpoints

The API provides the following endpoints:

- `POST /predict`: Accepts a JSON payload with an `excerpt` field containing the text to predict readability for. Returns a prediction score.

Example request:

```json
POST /predict
Content-Type: application/json

{
  "excerpt": "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape. The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches. At each end of the room, on the wall, hung a beautiful bear-skin rug. These rugs were for prizes, one for the girls and one for the boys. And this was the game. The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole. This would have been an easy matter, but each traveller was obliged to wear snowshoes."
}
```