# python-forceatlas2

An implementation of the ForceAtlas2 algorithm originally created for Gephi, 
ported from Java and implementing all of the features contained in the 
original paper (http://bit.ly/29DRQwe). 

This class requires numpy and python 2.6+.

Original java code can be found here: http://bit.ly/2azXlsj  
A similar attempt to port ForceAtlas2 can be found here: http://bit.ly/2aLjDGA  
- The other port does not implement all features, such as multithreading or avoid collision  

## Performance & Implementation
- Native C binary: Parallel repulsion & Barnes-Hut quadtree via `pthread` (~500x faster than pure Python).
- Falls back to vectorized engine with zero C compiler requirement (~6x faster than pure Python).
- Implements all features from the original paper: Barnes-Hut regional optimization, anti-collision node sizing, lin-log mode, hub dissuasion (outbound attraction distribution), and strong gravity.
- Run `python3 test_visual.py` to generate SVG layouts and an interactive timeline viewer (`layout_demo.html`).