import mesa
import numpy as np
import logging
import math

from decimal import Decimal


class Trace(Agent):
    def __init__(self, model):
        self.age = model.initialAge
        model.traces += 1
        super().__init__(model.traces, model)

    def step(self):
        self.age -= 1
        if self.age < 0:
            self.model.schedule.remove(self)
            self.model.space.remove_agent(self)


class Boid(mesa.Agent):
    """
    A Boid-style flocker agent. Aligned to the Netlogo flocking model [1]

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and velocity (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.

    [1] Wilensky, U. (1998). NetLogo Flocking model.
    http://ccl.northwestern.edu/netlogo/models/Flocking.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.

    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed,
        velocity,
        vision,
        separation,
        cohere=0.025,
        separate=0.25,
        match=0.04,
        headingx=0.0,
        headingy=0.0,
    ):
        """
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
            cohere: the relative importance of matching neighbors' positions
            separate: the relative importance of avoiding close neighbors
            match: the relative importance of matching neighbors' headings
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.velocity = velocity
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.heading = 0
        self.headingx = headingx
        self.headingy = headingy

        self.log = logging.getLogger("boid_" + str(self.unique_id))

    def correctAngle(self, angle):

        if angle < -180:
            angle += 360
        elif angle > 180:
            angle -= 360
        return angle

    def adjustAngle(self, angle, maxAngle, away=False):
        if away:
            newHeading = self.heading - angle
        else:
            newHeading = angle - self.heading

        self.correctAngle(newHeading)

        return min(abs(newHeading), maxAngle) * np.sign(newHeading)

    def getAngle(self, x, y):

        if x == 0:
            return 90 if y > 0 else -90
        elif y == 0:
            return 0 if x > 0 else 180
        elif x < 0:
            return -math.atan(y / x) / math.pi * 180
        else:
            return math.atan(y / x) / math.pi * 180

    def separate(self, nearestNeighbor):
        """
        Return a vector away from any neighbors closer than separation dist.
        """
        self.heading += self.adjustAngle(
            nearestNeighbor.heading, self.model.max_separate_angle, away=True
        )

    def cohere(self, neighbors):
        """
        Return the vector toward the center of mass of the local neighbors.
        """
        cohere = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                cohere += self.model.space.get_heading(self.pos, neighbor.pos)
            cohere /= len(neighbors)

        self.heading += self.adjustAngle(
            self.getAngle(cohere[0], cohere[1]), self.model.max_cohere_angle, away=False
        )

    def align(self, neighbors):
        """
        Return a vector of the neighbors' average heading.
        """
        match_vector = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                match_vector += (neighbor.headingx, neighbor.headingy)
                self.model.log.debug(
                    str(self.unique_id)
                    + ": "
                    + str(neighbor.unique_id)
                    + "> "
                    + str(match_vector)
                )
            match_vector /= len(neighbors)
        self.heading += self.adjustAngle(
            self.getAngle(match_vector[0], match_vector[1]),
            self.model.max_align_angle,
            away=False,
        )

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """
        nearestNeighbor = None
        distance = Decimal("Infinity")
        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        boids = set()
        for agent in neighbors:
            if isinstance(agent, Boid):
                boids.add(agent)
                tempDistance = self.model.space.get_distance(self.pos, agent.pos)
                if tempDistance < distance:
                    nearestNeighbor = agent
                    distance = tempDistance

        if nearestNeighbor:
            if distance < self.model.minimum_separation:
                oldHeading = self.heading
                self.separate(nearestNeighbor)
                self.log.debug(
                    str(self.unique_id) + ": Heading separate: %s > %s",
                    oldHeading,
                    self.heading,
                )
            else:
                oldHeading = self.heading
                self.align(boids)
                self.log.debug(
                    str(self.unique_id) + ": Heading align: %s > %s",
                    oldHeading,
                    self.heading,
                )
                oldHeading = self.heading
                self.cohere(boids)
                self.log.debug(
                    str(self.unique_id) + ": Heading cohere: %s > %s",
                    oldHeading,
                    self.heading,
                )

        # normalise heading
        self.setHeading(self.heading)
        distance = math.sqrt(self.headingx**2 + self.headingy**2)
        new_pos = (
            self.pos + np.array([self.headingx, self.headingy]) / distance * self.speed
        )
        if not self.model.toroidal:
            if self.model.space.out_of_bounds(new_pos):
                self.headingx *= -1
                self.headingy *= -1
                self.heading = self.correctAngle(self.heading - 180)
                new_pos = (
                    self.pos
                    + np.array([self.headingx, self.headingy]) / distance * self.speed
                )

        self.model.space.move_agent(self, new_pos)
        if self.model.traceAgent:
            self.trace()

    def setHeading(self, heading):
        # tan is not defined for Pi/2 and 3Pi/2:
        if heading == 90:
            self.headingx = 0
            self.headingy = 1
        elif heading == 270:
            self.headingx = 0
            self.headingy = -1
        # going backward:
        elif self.heading < -90 or self.heading > 90:
            self.headingx = -1
            self.headingy = -math.tan(math.pi / 180 * heading)
        # going forward
        else:
            self.headingx = 1
            self.headingy = math.tan(math.pi / 180 * heading)

        self.log.debug(
            str(self.unique_id) + ": " + str(self.heading) + " - " + str(self.headingy)
        )

    def trace(self):
        trace = Trace(self.model)
        self.model.space.place_agent(trace, self.pos)
        self.model.schedule.add(trace)
