from fastapi import FastAPI, File, UploadFile
from load_data import load_and_preprocess_image, load_and_preprocess_image_file
from model import load_model
from predict import predict_digit
from pydantic import BaseModel


class ImagePath(BaseModel):
    """
    Path to an image file.

    Attributes:
        path_to_img (str): Path to the image file.
    """

    path_to_img: str


app = FastAPI()

# Load the saved model
model = load_model("models/mnist_model")


@app.get("/")
def read_root():

    return {"Hello": "World"}


@app.get("/hello/{name}")
def hello_name(name: str):

    return {"Hello": name}


@app.post("/predict/")
def predict(image_path: ImagePath):


    image = load_and_preprocess_image(image_path.path_to_img)
    predicted_digit, confidence = predict_digit(model, image)

    return {"prediction": int(predicted_digit), "confidence": int(confidence * 100)}


@app.post("/predict_file/")
def predict_file(image_file: UploadFile = File(...)):

    # Load and preprocess the input image
    image = load_and_preprocess_image_file(image_file.file)
    predicted_digit, confidence = predict_digit(model, image)
    return {"prediction": int(predicted_digit), "confidence": int(confidence * 100)}


# if __name__ == '__main__':
#     """
#     Run the API server.
#     """

#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000)