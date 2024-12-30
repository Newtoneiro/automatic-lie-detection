# Zastosowanie metod uczenia maszynowego do automatycznej detekcji kłamstwa

## Opis projektu

Celem pracy jest opracowanie systemu automatycznego rozpoznawania kłamstwa na podstawie automatycznej analizy twarzy. W ramach pracy dyplomant powinien dokonać przeglądu literatury dotyczącej obecnych rozwiązań i technologii związanych z algorytmami detekcji kłamstwa na danych obrazowych oraz dokonać oceny zalet i wad istniejących algorytmów. Finalnie, wybrany algorytm powinien zostać zaimplementowany na podstawie artykułu źródłowego oraz porównany do innych znanych z rozwiązań literatury. Ewaluacja algorytmu powinna zostać wykonana na co najmniej jednej znanej z literatury bazie, np. Silenian Deception Dataset lub Miami University Deception Detection Database.

## Struktura Projektu:
    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
    ├── config  <- any configuration files
    ├── data
    │   ├── interim <- data in intermediate processing stage
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── docs  <- usage documentation or reference papers
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src
        ├── data-proc <- scripts for processing data eg. transformations, dataset merges etc. 
        ├── viz  <- scripts for visualisation during EDA, modelling, error analysis etc. 
        ├── modeling    <- scripts for generating models
    |--- environment.yml <- file with libraries and library versions for recreating the analysis environment
   
## Użyteczne linki:

VideoAnnotator -> https://github.com/roboflow/supervision/blob/0.22.0/examples/traffic_analysis/ultralytics_example.py

GOOGLE FACE MARKERER -> https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
                     -> https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
                     -> https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

EKSPRESJE -> https://pixabay.com/videos/search/facial%20expressions%20smile/

DATASET EMOCJI -> https://zenodo.org/records/1188976 | 1) Neutral, 2) Calm 3) Happy , 4) Sad, 5) Angry, 6) Fearful, 7) Disgust and 8) Surprised.