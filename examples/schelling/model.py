import mesa


class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent
    """

    def __init__(self, model: mesa.Model, agent_type: int) -> None:
        """
        Create a new Schelling agent.

        Args:
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(model)
        self.type = agent_type

    def step(self) -> None:
        neighbors = self.model.grid.iter_neighbors(
            self.pos, moore=True, radius=self.model.radius
        )
        similar = sum(1 for neighbor in neighbors if neighbor.type == self.type)

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(
        self,
        height=20,
        width=20,
        homophily=3,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        seed=None,
    ):
        """
        Create a new Schelling model.

        Args:
            width, height: Size of the space.
            density: Initial Chance for a cell to populated
            minority_pc: Chances for an agent to be in minority class
            homophily: Minimum number of agents of same class needed to be happy
            radius: Search radius for checking similarity
            seed: Seed for Reproducibility
        """

        super().__init__(seed=seed)
        self.homophily = homophily
        self.radius = radius

        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        self.happy = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={"happy": "happy"},  # Model-level count of happy agents
        )

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for _, pos in self.grid.coord_iter():
            if self.random.random() < density:
                agent_type = 1 if self.random.random() < minority_pc else 0
                agent = SchellingAgent(self, agent_type)
                self.grid.place_agent(agent, pos)

        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

        self.running = self.happy != len(self.agents)
