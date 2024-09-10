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
   
