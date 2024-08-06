import os
import sys
from io import StringIO

import cv2
import typer
import logging
from pyfiglet import Figlet

from pyrootmancer.scripts.features.instance_segmentation import InstanceSegmentation
from pyrootmancer.scripts.features.root_coord_extraction import LandmarkExtraction
from pyrootmancer.scripts.data.data_preprocessing import DataPipelineSetup
from pyrootmancer.scripts.models.model_training import ModelTraining
from pyrootmancer.scripts.utils.configuration import *


logging.basicConfig(level=logging.INFO, format='%(message)s')

output = sys.stdout
sys.stdout = captured_output = StringIO()
app = typer.Typer()


@app.command()
def pyrootmancer(args: str = typer.Argument(...)):
    if args == "go":
        ascii_art = Figlet(font='slant')
        logging.info(ascii_art.renderText("PyRootMancer"))


@app.command()
def create_folders():
    processor = DataPipelineSetup()
    processor.create_folders()


@app.command()
def unzip():
    processor = DataPipelineSetup()
    if os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "train.zip")):
        processor.unzip("train")
    else:
        logging.error("train.zip is not in the raw folder unzipping cannot be done")
        logging.error("make sure to add the zip in the raw data folder")
    if os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "train.zip")):
        processor.unzip("masks")
    else:
        logging.error("masks.zip is not in the raw folder unzipping cannot be done")
        logging.error("make sure to add the zip in the raw data folder")

    if os.path.exists(os.path.join(folder_config.get("raw_data_folder"), "train.zip")):
        processor.unzip("test")
    else:
        logging.error("test.zip is not in the raw folder unzipping cannot be done")
        logging.error("make sure to add the zip in the raw data folder")


@app.command()
def crop(args: str = typer.Argument(..., help="Choose between images, masks, or train (both)")):

    processor = DataPipelineSetup()
    if args == "images":
        processor.crop(folder_config.get("images_folder_unpatched"))
        logging.info("If you want to train a model you need to also crop the masks")
        logging.info("Do you want to crop the masks as well? (yes/no): ")
        outputs = input()
        if outputs.lower() == "yes":
            processor.crop(folder_config.get("root_folder_unpatched"))
        else:
            logging.warning("Both masks and image should be cropped in order to proceed training operation")

    if args == "masks":
        processor.crop(folder_config.get("root_folder_unpatched"))
        logging.info("If you want to train a model you need to also crop the images")
        logging.info("Do you want to crop the images as well? (yes/no): ")
        outputs = input()
        if outputs.lower() == "yes":
            processor.crop(folder_config.get("images_folder_unpatched"))
        else:
            logging.warning("Both masks and image should be cropped in order to proceed training operation")

    elif args == "train":
        processor.crop(folder_config.get("images_folder_unpatched"))
        processor.crop(folder_config.get("root_folder_unpatched"))


@app.command()
def patchify(args: str = typer.Argument(..., help="Choose between images, masks, or train (both)")):
    processor = DataPipelineSetup()
    if args == "images":
        processor.img_patchify(folder_config.get("images_folder_unpatched"), folder_config.get("images_folder_patched"))
        logging.info("If you want to train a model you need to also patchify the masks")
        logging.info("Do you want to patchify the masks as well? (yes/no): ")
        outputs = input()
        if outputs.lower() == "yes":
            processor.img_patchify(folder_config.get("root_folder_unpatched"), folder_config.get("root_folder_patched"))
        else:
            logging.warning("Both masks and image should be patchified in order to proceed training operation haha")

    if args == "masks":
        processor.img_patchify(folder_config.get("root_folder_unpatched"), folder_config.get("root_folder_patched"))
        logging.info("If you want to train a model you need to also patchify the images")
        logging.info("Do you want to patchify the images as well? (yes/no): ")
        outputs = input()
        if outputs.lower() == "yes":
            processor.img_patchify(
                folder_config.get("images_folder_unpatched"), folder_config.get("images_folder_patched")
            )
        else:
            logging.warning("Both masks and image should be patchified in order to proceed training operation")

    elif args == "train":
        processor.img_patchify(folder_config.get("images_folder_unpatched"), folder_config.get("images_folder_patched"))
        processor.img_patchify(folder_config.get("root_folder_unpatched"), folder_config.get("root_folder_patched"))


@app.command()
def data_checker():

    modelling = ModelTraining()
    logging.info("Check if images and masks have equal instances of classes")
    modelling.data_generator(
        os.path.dirname(folder_config.get("images_folder_patched")),
        os.path.dirname(folder_config.get("root_folder_patched")),
    )

    sys.stdout = output
    output_string = captured_output.getvalue()

    lines = output_string.strip().split('\n')
    found_lines = [line for line in lines if line.startswith('Found ')]

    if len(found_lines) >= 4:
        first_number = int(found_lines[0].split()[1])
        second_number = int(found_lines[1].split()[1])
        third_number = int(found_lines[2].split()[1])
        fourth_number = int(found_lines[3].split()[1])

    if first_number == third_number:
        typer.echo(
            f"{typer.style(str(first_number), fg=typer.colors.GREEN)} patches images and {typer.style(str(third_number), fg=typer.colors.GREEN)} patches masks are equal, ready for training"
        )
    else:
        typer.echo(
            f"{typer.style(str(first_number), fg=typer.colors.RED)} patches images != {typer.style(str(third_number), fg=typer.colors.RED)} patches masks cannot proceed training, check your data set"
        )
    if second_number == fourth_number:
        typer.echo(
            f"{typer.style(str(second_number), fg=typer.colors.GREEN)} patches images and {typer.style(str(fourth_number), fg=typer.colors.GREEN)} patches masks are equal, ready for validation"
        )
    else:
        typer.echo(
            f"{typer.style(str(second_number), fg=typer.colors.RED)} patches images != {typer.style(str(fourth_number), fg=typer.colors.RED)} patches masks cannot proceed training, check your data set"
        )


@app.command()
def train(
    type_train: str = typer.Argument("checkpoint", help="Train from checkpoint or from scratch"),
    epochs: int = typer.Argument(5, help="Enter the number of epochs (recommended between 5-15)"),
):

    modelling = ModelTraining()

    if type_train.lower() == "checkpoint":
        if epochs <= 0:
            logging.error("Epochs must be a positive integer.")
            return

        logging.info("Type training set to checkpoint")
        logging.info(
            f"Retraining existing model {typer.style('best_root_model', fg=typer.colors.BLUE)} is set to train with {typer.style(str(epochs), fg=typer.colors.BLUE)} epochs and default parameters"
        )
        modelling.training(
            int(epochs),
            os.path.dirname(folder_config.get("images_folder_patched")),
            os.path.dirname(folder_config.get("root_folder_patched")),
            folder_config.get("models_folder"),
            "best_root_model",
        )

    if type_train.lower() == "scratch":
        model_name = typer.prompt("Define the name of the new model (example: mymodel)", default="mymodel")
        if epochs <= 0:
            logging.error("Epochs must be a positive integer.")
            return

        logging.info("Choose optimizer adam, sgd, or rmsprop (adam is recommended")
        optimizer = typer.prompt("Choose optimizer adam, sgd, or rmsprop (adam is recommended))", type=str)
        logging.info("Choose patience for early stopping (3 is recommended)")
        patience = typer.prompt("Choose patience for early stopping (3 is recommended)", type=int)

        logging.info(
            f"Model {typer.style(model_name, fg=typer.colors.BLUE)} is set to train with {typer.style(str(epochs), fg=typer.colors.BLUE)} epochs and input parameters"
        )
        modelling.training(
            int(epochs),
            os.path.dirname(folder_config.get("images_folder_patched")),
            os.path.dirname(folder_config.get("root_folder_patched")),
            folder_config.get("models_folder"),
            str(model_name),
            int(patience),
            str(optimizer),
        )


@app.command()
def predict():
    modelling = ModelTraining()
    segmentation = InstanceSegmentation()
    available_models = [os.path.basename(model)[:-6] for model in os.listdir(folder_config.get("models_folder"))]
    if not available_models:
        logging.error("No models found")
        logging.error("First train model before proceeding predicting")
        return

    models_list = "\n".join(
        [f"{typer.style(i+1, fg=typer.colors.BLUE)} {model}" for i, model in enumerate(available_models)]
    )
    logging.info(f"Choose which model you want to use for predicting:\n{models_list}\nEnter the number of the model")
    choice = typer.prompt(
        f"Choose which model you want to use for predicting:\n{models_list}\nEnter the number of the model", type=int
    )
    choice = int(choice)
    try:
        if 1 <= choice <= len(available_models):
            chosen_model = available_models[choice - 1]
            logging.info(chosen_model)
            modelling.predict_folder(
                folder_config.get("test_folder"),
                folder_config.get("data_predictions_raw"),
                folder_config.get("models_folder"),
                chosen_model,
            )
            segmentation.return_original_size_folder(
                folder_config.get("data_predictions_raw"), folder_config.get("data_predictions_clean")
            )
            logging.info(
                f"Predicted masks saved to and stored in {os.path.basename(folder_config.get('data_predictions_raw'))}"
            )
        else:
            logging.error("Invalid choice. Please try again")
            return
    except ValueError:
        logging.error("Invalid input. Please enter a number")


@app.command()
def overlay():
    segmentation = InstanceSegmentation()
    segmentation.overlay(
        folder_config.get("test_folder"), folder_config.get("data_predictions_clean"), folder_config.get("root_labels")
    )
    labels = [os.path.basename(label)[:-4] for label in os.listdir(folder_config.get("root_labels"))]
    labels_list = "\n".join([f"{typer.style(i+1, fg=typer.colors.BLUE)} {label}" for i, label in enumerate(labels)])
    logging.info(f"\nChoose which test images to display predicted label:\n{labels_list}\nEnter the index of the image")
    choice = typer.prompt(
        f"\nChoose which test images to display predicted label:\n{labels_list}\nEnter the index of the image", type=int
    )

    choice = int(choice)
    try:
        if 1 <= choice <= len(labels):
            chosen_images = labels[choice - 1]
            logging.info(chosen_images)
            display_img = cv2.imread(os.path.join(folder_config.get("root_labels"), f"{chosen_images}.png"))
            cv2.imshow('predicted root', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            logging.error("Invalid choice. Please try again")
        return
    except ValueError:
        logging.error("Invalid input. Please enter a number")


@app.command()
def detect():
    landmarks = LandmarkExtraction()

    labels = [os.path.basename(label)[:-4] for label in os.listdir(folder_config.get("test_folder"))]
    labels_list = "\n".join([f"{typer.style(i+1, fg=typer.colors.BLUE)} {label}" for i, label in enumerate(labels)])
    logging.info(
        f"\nChoose which test images to display predicted root tips:\n{labels_list}\nEnter the index of the image"
    )
    choice = typer.prompt(
        f"\nChoose which test images to display predicted root tips:\n{labels_list}\nEnter the index of the image:",
        type=int,
    )
    choice = int(choice)
    try:
        if 1 <= choice <= len(labels):
            chosen_images = labels[choice - 1]
            image = landmarks.detect(
                chosen_images, folder_config.get("data_predictions_clean"), folder_config.get("test_folder"), choice
            )
            logging.info(f"\nDo you want to save this images? {chosen_images}\n(yes/no):")
            save = typer.prompt(f"\nDo you want to save this images? {chosen_images}\n(yes/no):")
            if save == "yes":
                cv2.imwrite(os.path.join(folder_config.get("root_landmarks"), f"{chosen_images}.png"), image)
                logging.info(f"Images saved successfully in {folder_config.get('root_landmarks')}")
                return
            else:
                return
        else:
            logging.error("Invalid choice. Please try again")
            return
    except ValueError:
        logging.error("Invalid input. Please enter a number")


if __name__ == "__main__":
    app()
