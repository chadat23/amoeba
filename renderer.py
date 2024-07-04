import random

from things import Thing, Amoeba, World

def rgb_ansi_bg(r, g, b):
    return f'\033[48;2;{r};{g};{b}m'


def render_world(world):
    multiplier = 2
    """array = [['*' * multiplier for _ in range(world.width + 2)]] + \
        [['*' * multiplier,] + [' ' * multiplier for _ in range(world.width)] + ['*' * multiplier,] for _ in range(world.height)] + \
        [['*' * multiplier for _ in range(world.width + 2)]]"""
    array = [[f'{i:>2}' for i in range(world.width + 2)]] + \
        [[f'{i:>2}',] + ['  ' for _ in range(world.width)] + [f'{i:>2}',] for i in range(world.height)] + \
        [[f'{i:>2}' for i in range(world.width + 2)]]
    
    for thing in world.things:
        for i in range(thing.x + 1, thing.x + thing.width + 1):
            for j in range(thing.y + 1, thing.y + thing.height + 1):
                array[j][i] = rgb_ansi_bg(*thing.smell) + ' ' * multiplier + '\033[0m'

    amoeba_smell = [[[0, 0, 0] for _ in range(world.amoeba.width)] for _ in range(world.amoeba.height)]
    for thing in world.things:
        smell = world.amoeba.smell_from_area(thing.smell, thing.x, thing.y, thing.width, thing.height)
        for i in range(len(amoeba_smell)):
            amoeba_smell[i] = smell[i]

    for i in range(len(amoeba_smell)):
        for j in range(len(amoeba_smell[0])):
            for k in range(len(amoeba_smell[0][0])):
                amoeba_smell[i][j][k] = int(amoeba_smell[i][j][k] / 1)

    print(amoeba_smell)
    for i in range(world.amoeba.x + 1, world.amoeba.x + world.amoeba.width + 1):
        for j in range(world.amoeba.y + 1, world.amoeba.y + world.amoeba.height + 1):
            array[j][i] = rgb_ansi_bg(*amoeba_smell) + 'X' * multiplier + '\033[0m'

    for row in array:
        print(''.join(row))

if __name__ == "__main__":
    """cols = 16
    rows = 16
    amoeba_width = 3
    amoeba_height = 3
    amoeba = Amoeba(random.randrange(0, cols - amoeba_width), random.randrange(0, rows - amoeba_height), amoeba_width, amoeba_height, [0, 0, 0])
    thing_width = 2
    thing_height = 2
    thing0 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    thing1 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    thing2 = Thing(random.randrange(0, cols - thing_width), random.randrange(0, rows - thing_height), thing_width, thing_height, [12, 58, 90])
    world = World(cols, rows, amoeba, [thing0, thing1, thing2])"""
    cols = 16
    rows = 16
    amoeba_width = 3
    amoeba_height = 3
    amoeba = Amoeba(8, 13, amoeba_width, amoeba_height, [0, 0, 0])
    thing_width = 2
    thing_height = 2
    thing0 = Thing(3, 6, thing_width, thing_height, [255, 58, 90])
    thing1 = Thing(10, 1, thing_width, thing_height, [12, 255, 90])
    thing2 = Thing(12, 10, thing_width, thing_height, [12, 58, 255])
    world = World(cols, rows, amoeba, [thing0, thing1, thing2])

    render_world(world)
