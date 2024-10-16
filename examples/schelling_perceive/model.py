import mesa

from mesa.space import Coordinate, SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import numpy as np


class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, pos, model, agent_type):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           x, y: Agent initial location.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.happy = 0

    def step(self):
        similar = 0
        all = 0

        for neighbor in self.model.grid.iter_neighbors(
            self.pos,
            moore=self.model.perception_moore,
            radius=self.model.perception_radius,
        ):
            all += 1
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if all == 0 or similar / all < self.model.homophily:
            self.model.grid.move_to_empty(self)
            self.model.moves +=1
            self.happy = 0
        else:
            self.model.happy += 1
            self.happy = 1



class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    NEIGHBOURHOOD_MOORE = "Moore"
    NEIGHBOURHOOD_VON_NEUMANN = "von Neumann"

    def __init__(
        self,
        width=20,
        height=20,
        density=0.8,
        minority_pc=0.2,
        homophily=3 / 8,
        perception_neighbourhood="Moore",
        perception_radius=1,
        seed=0,
    ):
        """ """
        super().__init__()
        
        np.random.seed(seed)

        self.width = width
        self.height = height
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)

        self.happy = 0
        self.moves = 0
        self.datacollector = DataCollector(
            {"happy": "happy", "moves": "moves"},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            # {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )

        self.perception_moore = (
            True if perception_neighbourhood == Schelling.NEIGHBOURHOOD_MOORE else False
        )
        self.perception_radius = perception_radius

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                if self.random.random() < self.minority_pc:
                    agent_type = 1
                else:
                    agent_type = 2

                agent = SchellingAgent(pos, self, agent_type)
                self.grid.move_agent(agent, pos)
                self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        if self.happy == self.schedule.get_agent_count():
            self.running = False

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()
