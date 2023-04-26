import mesa

from .model import BoidFlockers
from .boid import Boid, Trace
from .SimpleContinuousModule import SimpleCanvas
from mesa.visualization.UserParam import UserSettableParameter


def boid_draw(agent):
    if type(agent) is Boid:
        return {
            "Shape": "circle",
            "r": 5,
            "Filled": "true",
            "Layer": 2,
            "Color": "red",
            "text": agent.unique_id,
            "text_color": "black",
            "scale": 0.8,
        }
    if type(agent) is Trace:
        return {
            "Shape": "circle",
            "r": 1,
            "Filled": "true",
            "Layer": 3,
            "Color": "#" + "".join([str(9 - agent.age) for _ in range(6)]),
        }


boid_canvas = SimpleCanvas(boid_draw, 500, 500)

model_params = {
    "seed": UserSettableParameter("number", "Random seed", value=0),
    "population": UserSettableParameter(
        "slider", "Population size", value=100, min_value=10, max_value=1000, step=10
    ),
    "width": UserSettableParameter(
        "slider", "Grid width", value=100, min_value=10, max_value=1000, step=10
    ),
    "height": UserSettableParameter(
        "slider", "Grid height", value=100, min_value=10, max_value=1000, step=10
    ),
    "toroidal": UserSettableParameter("checkbox", "Toroidal?", value=True),
    "traceAgent": UserSettableParameter("checkbox", "trace agents?", value=False),
    "initialAge": UserSettableParameter(
        "slider", "Trace length", value=5, min_value=1, max_value=9, step=1
    ),
    "speed": UserSettableParameter(
        "slider", "Speed", value=5, min_value=1, max_value=20, step=1
    ),
    "minimum_separation": UserSettableParameter(
        "slider",
        "Minimum separation",
        value=0.5,
        min_value=0.1,
        max_value=2.0,
        step=0.01,
    ),
    "max_separate_angle": UserSettableParameter(
        "slider",
        "Max separation angle",
        value=1.5,
        min_value=0.5,
        max_value=20.0,
        step=0.5,
    ),
    "max_cohere_angle": UserSettableParameter(
        "slider",
        "Max coherence angle",
        value=3.0,
        min_value=0.5,
        max_value=20.0,
        step=0.5,
    ),
    "max_align_angle": UserSettableParameter(
        "slider", "Max align angle", value=5.0, min_value=0.5, max_value=20.0, step=0.5
    ),
    "vision": UserSettableParameter(
        "slider", "Vision", value=10, min_value=1, max_value=30, step=1
    ),
}

model_params["population"].value = 20
model_params["speed"].value = 2
model_params["traceAgent"].value = True

server = mesa.visualization.ModularServer(
    BoidFlockers, [boid_canvas], "Boids", model_params
)
