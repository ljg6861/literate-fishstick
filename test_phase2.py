import sys
import traceback
from simulation.world import World
from config import CONFIG
import os

try:
    print("Testing World initialization...")
    w = World(1920, 1080, CONFIG)
    w.initialize()
    print("Init OK")
    
    print("Running 100 ticks...")
    for i in range(100):
        w.tick_simulation()
        if i % 10 == 0:
            print(f"Tick {i} ok")
            
    print("Done.")

except Exception:
    traceback.print_exc()
