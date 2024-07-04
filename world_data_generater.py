import random
import csv

import time
import os
import colorama
from colorama import Fore, Back, Style

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
        total = [[[0, 0, 0] for _ in range(self.width)] for _ in range(self.height)]

        for i in range(x, x + width):
            for j in range(y, y + height):
                if not (self._x <= i <= self._x + self._width and self._y <= j <= self._y + self._height):
                    for m in range(self._x, self._x + self._width):
                        for n in range(self._y, self._y + self._height):
                            rad = ((i - m)**2 + (j - n)**2)**0.5
                            for k in range(len(total)):
                                total[m - self._x][n _ self._y][k] += (smell[k] / rad**)
                                #total[k] += (smell[k] / rad**2)
    
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
    pass

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

def generate_world(rows=8, cols=8):
    amoeba_width = 3
    amoeba_height = 3
    amoeba = Amoeba(random.randrange(0, cols - amoeba_width), random.randrange(0, rows - amoeba_height), amoeba_width, amoeba_height, [0, 0, 0])
    thing_width = 2
    thing_height = 2
    thing0 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    thing1 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    thing2 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    world = World(8, 8, amoeba, [thing0, thing1, thing2])

    return world

def flatten(array):
    """
            Flattens a 2D array into a 1D list.

            Parameters:
            array (list): 2D list to be flattened.

            Returns:
            list: Flattened 1D list.
            """
    return [item for sublist in array for item in sublist]

def save_arrays_to_file(filename, all_arrays):
    """
            Saves all sets of features and labels to a CSV file.

            Parameters:
            filename (str): The name of the file to save the arrays to.
        all_arrays (list): List of tuples containing 2D lists of features and labels.
        """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
                
        for arrays in all_arrays:
            features_flat = flatten(arrays[0])
            labels_flat = flatten(arrays[1])
            writer.writerow(features_flat + labels_flat)

if __name__ == "__main__":
    rows = 8
    cols = 8
    runs = 10
    filename = "arrays_output.csv"
    all_worlds = []

    #for _ in range(runs):
    i = 0
    while i < runs:
        world = generate_world(rows, cols)
        #all_worlds.append(world)
        if not world.has_intersections():
            all_worlds.append(world)
            i += 1

    for world in all_worlds:
        smell = [0, 0, 0]
        for thing in world.things:
            thing_smell = world.amoeba.smell_from_area(thing.smell, thing.x, thing.y, thing.width, thing.height)
            for i in range(len(smell)):
                smell[i] += thing_smell[i]
            print(f"smell {smell}")


        print(f"Amoeba: x {world.amoeba.x:>2}, y: {world.amoeba.y:>2}, width: {world.amoeba.width:>2}, height: {world.amoeba.height:>2}")
        for count, thing in enumerate(world.things):
            print(f"Thing{count}: x {thing.x:>2}, y: {thing.y:>2}, width: {thing.width:>2}, height: {thing.height:>2}")
    #save_arrays_to_file(filename, all_arrays)

