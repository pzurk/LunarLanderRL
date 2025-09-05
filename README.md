## Struktura projektu

### `src/` – implementacja agentów
- **LunarLanderDqnImage.py** – Agent DQN uczony na podstawie obrazu ze środowiska.
- **LunarLanderDqnSensor.py** – Agent DQN uczony na podstawie danych sensorycznych.
- **LunarLanderRainbowCamera.py** – Agent Rainbow uczony na podstawie obrazu z kamery.
- **LunarLanderRainbowImage.py** – Agent Rainbow uczony na podstawie obrazu ze środowiska.

### `test/` – skrypty do testowania i wizualizacji
- **PlotFromCSV.py** – Skrypt do wizualizacji zebranych danych podczas uczenia.
- **testLunarLanderDqnImage.py** – Testowanie modelu DQN opartego na obrazie.
- **testLunarLanderDqnSensor.py** – Testowanie modelu DQN opartego na danych sensorycznych.
- **testLunarLanderRainbowImage.py** – Testowanie modelu Rainbow opartego na obrazie.

### Wymagania
- `requirements.txt` – lista pakietów potrzebnych do uruchomienia aplikacji.
