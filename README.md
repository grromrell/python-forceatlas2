# python-forceatlas2

An implementation of the ForceAtlas2 algorithm originally created for Gephi, 
ported from Java and implementing all of the features contained in the 
original paper (http://bit.ly/29DRQwe). 

This class requires numpy and python 2.6+.

Original java code can be found here: http://bit.ly/2azXlsj  
A similar attempt to port ForceAtlas2 can be found here: http://bit.ly/2aLjDGA  
- This port does not implement all features, such as multiprocessing or avoid collision  
- This port does have cython optimization, making the base case faster then this version

Future Goals (to be accomplished as required):  
- Cythonize  
- Make into a module  
- Refactor to make more pythonic? (Currently a pretty direct Java port)
