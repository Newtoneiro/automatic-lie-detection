- Dostęp do bazy danych.
- Na jakim zaganieniu powinna skupić się praca? Detekcja autentyczności uśmiechu?
    - Opisać ogólnie problemy rozpoznawania kłamstw, liczy sie gestykulacja itd, ale na potrzeby pracy
    żeby skupić się na obrazie wideo wystarczy autentyczność uśmiechu (Pokrycie z bazami danych ~?)
    - Inny pomysł (zależy od bazy danych) - w kryminalistyce często ciężko odkryć prawdziwe intencje przesłuchiwanych, np. członek rodziny udający żałobę będąc tak na prawde zaangażowanym w morderstwo.
- W jakim stopniu mogę korzystać z zewnętrznych rozwiązań?
- Czy jakiś frontend do demnostracji / czy wystarczy działający algorytm i dokumentacja?
- Raczej będę używać templatki cookiecuttera.


### 10/17

- co poprawić w preprocesingu / processingu klatek
Face frontalization -> oczy zawsze w tym samym miejscu | poszukać prac związanych z tą normalizacją i znaleźć najnowszą / najlepszą

3 różne podejścia - przesunięcia punktów względem nosa / pozycje punktów ze znormalizowanymi oczami / czestotliwość mrugnięć (wykrywanie mrugnięć)

- jak bardzo skupiać się na porównaniach rzeczy, tak jak w inżynierce?
nie bardzo, liczy sie podejscie algorytmiczne, nie techniczne

- czy to już czas na wybór modelu - najwyrazniej nie

- czy dołączamy audio do modelu?
skupic sie na video, audio jest cieżkie.

### 24/10

- filmik rozłożyć na klatki, wziąć pierwsze x klatek, zrobić oś czasu i namalować wykres zależności (y - np. odległość oczu od nosa , x -klatka)
i na jego podstawie zweryfikować korelacje - podczas uśmiechu odleglosc kacików ust się zwieksza

- znormalizować twarz jako teksturę (frontalizacja obrazka, nie tylko punktów)
  