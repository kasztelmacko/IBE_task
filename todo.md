Step 1:
- Weryfikacja braków danych: sprawdzić czy Null jest tylko w kolumnach mat_ i oznacza że zadanie ni było w danym pomiarze
- Weryfikacja randomizacji: sprawdzić czy dana szkoła ma przypisaną tyko jedną grupę
- sprawdź czy są uczniowie co byli w preteście ale nie w postteście

Step 2:
- Total szkół i Total uczniów
- Macierz: liczba szkół i uczniów w podziale na grupe
- Dla każdego zadania w pomiarach obliczyć średni wynik (wskaźnik trudności). Sprawdzić czy są zadania esktremalnie łatwe / trudne, które nic nie wnoszą do modelu

Step 3:
- SPrawdzenie jednowymiarowości: przeprowadzenie analizy czynnikowej + scree plot. Chcemy sprawdzić czy zestaw zadań mierzy jeden ukryty konstrukt
- CTT jako punkt odniesienia: Oblicz Alfę Cronbacha dla obu pomiarów. Obliczyć korelacje pozycja-wynik, czyli moc dyskryminacyjną CTT
- Modelowanie IRT: wytreniuj 2PL i 3PL i porównać je. Analiza paremtrów modelu - sprawdzić alfe (dyskryminacje)
- Diagnostyka modelu: Item fit (czy poszczególne zadania pasują do modelu) + wykresy ICC dla kilku zadań 

Step 4:
- Estymiacja thety: przypisujemy każdemu uczniowi wartość ukrytą umiejętności matematycznych. Pewnie wykorzystamy EAP (Expected a posteriori). Pomiary są niezależne więc będziemy mieć thete dla każdego ucznia w każdym pomiarze
- Skalowanie i standaryzacja wyników: tu pewnie z-score bo Theta z IRT powinna mieć średnią 0 i std. 1 (do potwierdzenia)
- Statystyki opisowe thety dla pomiarów + weryfikacja rozkładu + Marginal Reliability (rzetelność empiryczna)
- Przygotowanie do weryfikacji hipotezy: połączyć oszacowane thety pre i post
- Analiza korelacji dla ucznia pre i post (silna korelacja mówi że testy mierzą ten sam konstrukt)

Step 5:
- Budowa modelu mieszanego: y - theta_post; fixed effects: grupa, theta_pre; random effect: id_szkoly (Lmer)
- Weryfikacja: ICC - współczynnik korelacji wewnątrzklasowej + normalność reszt + homoskedastyczność
- Interpretacja