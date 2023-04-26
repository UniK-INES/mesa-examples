"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np
import logging

from .boid import Boid


class BoidFlockers(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        seed=None,
        population=100,
        width=100,
        height=100,
        toroidal=False,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.025,
        separate=0.25,
        match=0.04,
        traceAgent=False,
        initialAge=5,
        minimum_separation=1.0,
        max_separate_angle=1.5,
        max_cohere_angle=3.0,
        max_align_angle=5.0,
    ):
        """
        Create a new Flockers model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separation: What's the minimum distance each Boid will attempt to
                    keep from any other
            cohere, separate, match: factors for the relative importance of
                    the three drives."""
        self.population = population
        self.vision = vision
        self.speed = speed
        self.separation = separation
        self.schedule = mesa.time.RandomActivation(self)
        self.toroidal = toroidal
        self.space = mesa.space.ContinuousSpace(width, height, toroidal)
        self.factors = dict(cohere=cohere, separate=separate, match=match)
        self.make_agents()
        self.running = True

        self.traceAgent = traceAgent
        self.initialAge = initialAge
        self.traces = population + 100

        self.max_separate_angle = max_separate_angle
        self.max_cohere_angle = max_cohere_angle
        self.max_align_angle = max_align_angle
        self.minimum_separation = minimum_separation

        self.log = logging.getLogger("boid")
        logging.basicConfig(
            level=logging.INFO, format=" %(asctime)s -%(levelname)s - %(message)s"
        )

        logger = logging.getLogger("boid_8")
        logger.setLevel(logging.DEBUG)

    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            velocity = np.random.random(2) * 2 - 1
            boid = Boid(
                i,
                self,
                pos,
                self.speed,
                velocity,
                self.vision,
                self.separation,
                **self.factors
            )
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        self.schedule.step()
