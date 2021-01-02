from src.training import start_training






if __name__ == '__main__':
    CSV_FILE = r"C:\Users\kicjo\Documents\Repositories\videoForCV\torch_light\casava_data\train.csv"
    JSON_FILE = r"C:\Users\kicjo\Documents\Repositories\videoForCV\torch_light\casava_data\label_num_to_disease_map.json"
    IMAGES_DIR = r"C:\Users\kicjo\Documents\Repositories\videoForCV\torch_light\casava_data\train_images"
    TEST_IMAGES_DIR = r"C:\Users\kicjo\Downloads\cassava-leaf-disease-classification\sample"
    # TRAINING_EPOCHS= 1
    # FINE_TUNING_EPOCHS= 1
    # IMG_SIZE = 28
    # LEARNING_RATE=2e-4
    # VIZ_DATA = False
    # BATCH_SIZE=64

    TRAINING_EPOCHS= 0
    FINE_TUNING_EPOCHS= 1
    IMG_SIZE = 70 #int(0.1 * self.input_height) must be odd
    LEARNING_RATE=2e-4
    VIZ_DATA = False
    BATCH_SIZE=8
    USE_GPU = False

    start_training(csv_file=CSV_FILE,
                   json_file=JSON_FILE,
                   images_dir=IMAGES_DIR,
                   test_images_dir=TEST_IMAGES_DIR,
                   training_max_epochs=TRAINING_EPOCHS,
                   finetuning_max_epochs=FINE_TUNING_EPOCHS,
                   img_size=IMG_SIZE,
                   learning_rate=LEARNING_RATE,
                   viz_datasets=VIZ_DATA,
                   batch_size=BATCH_SIZE,
                   use_gpus=USE_GPU,
                   )