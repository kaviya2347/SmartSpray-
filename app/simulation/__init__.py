"""
Drone mission simulator (backend side).

The 3D browser simulator (simulator/web/) flies the drone; this package plays
the part of the drone's camera and onboard AI:

    world.py    synthetic tomato field + camera image rendering
    service.py  mission steps: markers -> overview map -> per-plant capture
                -> route plan / re-plan -> flight summary
    router.py   /sim/* API and the /simulator web page
"""
