# LunarLanderRL

Research-oriented reinforcement learning project focused on comparing how agent performance changes with different learning algorithms and input in `LunarLander-v2`.

The core goal of the project was to evaluate how well agents can learn control policies from:

- low-dimensional sensor/state data,
- rendered frames from the environment,
- live camera input observing the screen in real time.

The most distinctive part of the project is the visual training pipeline based on a live camera image instead of state from environment.

## Experiment Variants

| Variant | Algorithm | Input |
| --- | --- | --- |
| `LunarLanderDqnSensor.py` | DQN | environment state vector |
| `LunarLanderDqnImage.py` | DQN + CNN | rendered environment frames |
| `LunarLanderRainbowImage.py` | Rainbow DQN | rendered environment frames |
| `LunarLanderRainbowCamera.py` | Rainbow DQN | live camera image |

## Tech Stack

- Python
- PyTorch
- Gymnasium
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Train selected variants:

```
python src/LunarLanderDqnSensor.py
python src/LunarLanderDqnImage.py
python src/LunarLanderRainbowImage.py
python src/LunarLanderRainbowCamera.py
```

Evaluate trained models:

```
python test/testLunarLanderDqnSensor.py
python test/testLunarLanderDqnImage.py
python test/testLunarLanderRainbowImage.py
```

Visualize training logs:

```
python test/PlotFromCSV.py training_data.csv --rewards --success
```