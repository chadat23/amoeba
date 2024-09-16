from copy import deepcopy
import random
import csv

from things import Thing, Amoeba, World


def generate_worlds(rows_min=12, cols_min=12, rows_max=30, cols_max=30, runs=10, max_things=8, filename="arrays_output.csv"):
    worlds = []

    i = 0
    while i < runs:
        rows = random.randrange(12, 30)
        cols = random.randrange(12, 30)
        amoeba_width = 3
        amoeba_height = 3
        amoeba = Amoeba(random.randrange(0, cols - amoeba_width), random.randrange(0, rows - amoeba_height), amoeba_width, amoeba_height, [0, 0, 0])
        thing_width = 1
        thing_height = 1
        things = []
        for _ in range(random.randrange(1, max_things)):
            things.append(Thing(random.randrange(0, cols - thing_width), \
                                random.randrange(0, rows - thing_height), \
                                thing_width, \
                                thing_height, \
                                [random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)]))
        world = World(rows, cols, amoeba, things)
        if world.has_intersections():
            continue
        world_small_change = deepcopy(world)
        if not world_small_change.move_thing_nonoverlaping_rand():
            continue
        good = True
        world_large_change = deepcopy(world)
        for _ in range(len(things)):
            if not world_large_change.move_thing_nonoverlaping_rand(2):
                good = False
                break
        if not good:
            break
        worlds.append([world, world_small_change, world_large_change])
        i += 1

    return worlds

def world_to_string(worlds):
    text = ''
    for world_set in worlds:
        for world in world_set:
            text += ','.join([str(v) for v in flatten_3d(world.amoeba.smell_from_things(world.things))]) + ','
        world = world_set[0]
        amoeba = world.amoeba
        text += f'{str(len(world.things))},'
        count = len(world.things)
        for t in world.things:
            text += ','.join([str(v) for v in [amoeba.x - t.x, amoeba.y - t.y, t.smell[0], t.smell[1], t.smell[2]]]) + ','
        text += ','.join(['0,0,0,0,0' for _ in range(8 - count)])
        text = text + '\n'

    return text

def flatten_2d(array):
    """
            Flattens a 2D array into a 1D list.

            Parameters:
            array (list): 2D list to be flattened.

            Returns:
            list: Flattened 1D list.
            """
    return [item for sublist in array for item in sublist]

def flatten_3d(array):
    """
            Flattens a 3D array into a 1D list.

            Parameters:
            array (list): 3D list to be flattened.

            Returns:
            list: Flattened 1D list.
            """
    return [elem for sublist1 in array for sublist2 in sublist1 for elem in sublist2]

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

def do_it(rows_min=12, cols_min=12, rows_max=30, cols_max=30, runs=10, max_things=8, filename="arrays_output.csv", should_print=False):
    worlds = generate_worlds(rows_min, cols_min, rows_max, cols_max, runs, max_things, filename)

    text = 'smell_0_R,smell_0_G,smell_0_B,' \
            'smell_1_R,smell_1_G,smell_1_B,' \
            'smell_2_R,smell_2_G,smell_2_B,' \
            'smell_3_R,smell_3_G,smell_3_B,' \
            'smell_4_R,smell_4_G,smell_4_B,' \
            'smell_5_R,smell_5_G,smell_5_B,' \
            'smell_6_R,smell_6_G,smell_6_B,' \
            'smell_7_R,smell_7_G,smell_7_B,' \
            'smell_8_R,smell_8_G,smell_8_B,' \
            'smell_small_0_R,smell_small_0_G,smell_small_0_B,' \
            'smell_small_1_R,smell_small_1_G,smell_small_1_B,' \
            'smell_small_2_R,smell_small_2_G,smell_small_2_B,' \
            'smell_small_3_R,smell_small_3_G,smell_small_3_B,' \
            'smell_small_4_R,smell_small_4_G,smell_small_4_B,' \
            'smell_small_5_R,smell_small_5_G,smell_small_5_B,' \
            'smell_small_6_R,smell_small_6_G,smell_small_6_B,' \
            'smell_small_7_R,smell_small_7_G,smell_small_7_B,' \
            'smell_small_8_R,smell_small_8_G,smell_small_8_B,' \
            'smell_large_0_R,smell_large_0_G,smell_large_0_B,' \
            'smell_large_1_R,smell_large_1_G,smell_large_1_B,' \
            'smell_large_2_R,smell_large_2_G,smell_large_2_B,' \
            'smell_large_3_R,smell_large_3_G,smell_large_3_B,' \
            'smell_large_4_R,smell_large_4_G,smell_large_4_B,' \
            'smell_large_5_R,smell_large_5_G,smell_large_5_B,' \
            'smell_large_6_R,smell_large_6_G,smell_large_6_B,' \
            'smell_large_7_R,smell_large_7_G,smell_large_7_B,' \
            'smell_large_8_R,smell_large_8_G,smell_large_8_B,' \
            'thing_count,' \
            'x0,y0,r0,g0,b0,' \
            'x1,y1,r1,g1,b1,' \
            'x2,y2,r2,g2,b2,' \
            'x3,y3,r3,g3,b3,' \
            'x4,y4,r4,g4,b4,' \
            'x5,y5,r5,g5,b5,' \
            'x6,y6,r6,g6,b6,' \
            'x7,y7,r7,g7,b7\n' \

    text += world_to_string(worlds)

    text = text[:-1]

    with open(filename, 'w') as file:
        file.write(text)

    if should_print:
        print(text)

if __name__ == "__main__":
    do_it(runs=10000, should_print=False, filename='training_data.csv')
    
