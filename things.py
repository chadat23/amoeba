import random

class Thing:
    def __init__(self, x, y, width, height, smell):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self.smell = smell

    def has_intersections(self, other):
        """if self.x <= other.x and other.x + other.width <= self.x + self.width and \
                self.y <= other.y and other.y + other.height <= self.y + self.height:
            print("intersect")
            return True
        """
        if (self.x <= other.x <= self.x + self.width or self.x <= other.x + other.width <= self.x + self.width or \
                other.x <= self.x <= other.x + other.width or other.x <= self.x + self.width <= other.x + other.width) and \
                (self.y <= other.y <= self.y + self.height or self.y <= other.y + other.height <= self.y + self.height or \
                other.y <= self.y <= other.y + other.height or other.y <= self.y + self.height <= other.y + other.height):
                    return True
        
        return False

    def smell_from_area(self, smell, x, y, width, height):
        total = [[[0, 0, 0] for _ in range(self.height)] for _ in range(self.width)]

        for i in range(x, x + width):
            for j in range(y, y + height):
                """if not (self._x <= i <= self._x + self._width and self._y <= j <= self._y + self._height):
                    for m in range(self._x, self._x + self._width):
                        for n in range(self._y, self._y + self._height):
                            rad = ((i - m)**2 + (j - n)**2)**0.5
                            for k in range(3):
                                total[m - self._x][n - self._y][k] += (smell[k] / rad**2)"""
                for m in range(self._x, self._x + self._width):
                    for n in range(self._y, self._y + self._height):
                        rad = ((i - m)**2 + (j - n)**2)**0.5
                        for k in range(3):
                            if rad != 0:
                                total[m - self._x][n - self._y][k] += (smell[k] / rad**2)
    
        return total

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @x.setter
    def x(self, value):
        self._x = value

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, value):
        self._width = value

    @height.setter
    def height(self, value):
        self._height = value

    def move_rand(self):
        self._x += random.randrange(-1, 2)
        self._y += random.randrange(-1, 2)

class Life(Thing):
    pass

class Amoeba(Life):
    def smell_from_things(self, things):
        amoeba_smell = [[[0, 0, 0] for _ in range(self.height)] for _ in range(self.width)]
        for thing in things:
            smell = self.smell_from_area(thing.smell, thing.x, thing.y, thing.width, thing.height)
            for i in range(len(amoeba_smell)):
                for j in range(len(amoeba_smell[0])):
                    for k in range(len(amoeba_smell[0][0])):
                        amoeba_smell[i][j][k] += smell[i][j][k]

        return amoeba_smell


class World():
    def __init__(self, width, height, amoeba, things):
        self.width = width
        self.height = height
        self.amoeba = amoeba
        self.things = things

    def has_intersections(self):
        for thing in self.things:
            if self.amoeba.has_intersections(thing):
                print("inter0")
                return True

        things_count = len(self.things)
        if things_count > 1:
            for i in range(things_count):
                for j in range(i + 1, things_count):
                    if self.things[i].has_intersections(self.things[j]):
                        print(f"{i}, {j} inter0")
                        return True

        return False

    def move_rand_amoeba(self):
        self.amoeba.move_rand()

    def solve(self):
        if self.amoeba.x < 0:
            self.amoeba.x = 0
        if self.amoeba.x > self.width:
            self.amoeba.x = self.width

        if self.amoeba.y < 0:
            self.amoeba.y = 0
        if self.amoeba.y > self.height:
            self.amoeba.y = self.height

