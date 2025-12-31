# automatic-pain-recognition
Bachelor thesis: Automatic pain regonition with the use of computer vision algorithms 

opis zamysłu projektu 
opis jak uruchamiac 
opis structury rozlozenia 


data 
    datasets
        BioVid.md
        RAVDESS.md
    frontalization

    landmarks
        top_100_
data preparation
    data_division
        data_division_BioVid
        data_division_RAVDESS
    data_preparation
        video into frames notebook 
        face_detection_notebook 
        point detection notebook (dlib and mediapipe)
        frontalization notebook (dlib and mediapipe)
        centering notebook
        dlib_mediapipe_comparions
        data_preparation_BIOVID
        data_preparation_RAVDESS   
    klasa z potokiem przetwarzania dlib
    klasa z potokiem przetwarzania mediapipe
    kalsa do przetworzenia danego zbioru danych który zostanie podany
verification of processing pipeline_emotion_detection
    veryfication of processed data notebooks
        veryfication_of_processed_data_BioVId
        veryfication_of_processed_data_Ravdess
        avarge_movements_of_points_BIOVID
        avargae_movements_of_points_RAVDESS
    training_emotions
        training_evaluation_478_points
        training_evaluation_100_points
        training_evaluation_478_movements
        training_evaluation_100_movements
        training_evaluation_100_movements_normalized
models
    BI-LSTM
    Attention LSTM
    Transformer

pain_detection
    training_notebooks
        training_BiLSTM_bianry
        training_BILSTM_multiclass
        training_AttentionLSTM_bianry
        training_AttentionLSTM_multiclass
        training_Transformer_bianry
        training_Transformer_multiclass

training_utils 
    train.py
    evaluate.py
    preprocess_dataset.py

mobile_app
    cały kod apki mobilnej 

serwer_for_mobile_app_code

venv
.gitignore
README.md
requirements.txt
PDI.pdf