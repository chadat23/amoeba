import time
import os

# Function to return ANSI escape sequence for 24-bit color
def rgb_ansi(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

# Define your RGB frames
frames = [
    ([255, 0, 0], "    *    \n   ***   \n  *****  \n ******* \n*********"),
    ([0, 255, 0], "    *    \n   ***   \n  *****  \n ******* \n*********"),
    ([0, 0, 255], "    *    \n   ***   \n  *****  \n ******* \n*********"),
    ([255, 255, 0], "    *    \n   ***   \n  *****  \n ******* \n*********"),
    ([0, 255, 255], "    *    \n   ***   \n  *****  \n ******* \n*********"),
    ([255, 0, 255], "    *    \n   ***   \n  *****  \n ******* \n*********"),
]

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def animate():
    while True:
        for rgb, text in frames:
            clear_console()
            color = rgb_ansi(*rgb)
            print(color + text + '\033[0m')  # Reset color at the end
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        animate()
    except KeyboardInterrupt:
        pass

