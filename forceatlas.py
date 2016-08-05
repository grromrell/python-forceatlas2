import random
import numpy as np
import multiprocessing

from queue import Queue
from threading import Thread

class ForceAtlas2:
    
    def __init__(self, graph, iterations, pos=None, sizes=None, directed=False, barnes_hut_theta=1.2, 
                 edge_weight_influence=0, gravity=0, jitter_tolerance=1, scaling_ratio=2, adjust_sizes=False, 
                 barnes_hut_optimize=False, lin_log_mode=False, outbound_attraction_distribution=False, 
                 strong_gravity_mode=False):

        """
        Class to faciliate force atlas iterations. Run_algo is the main 
        function that will perform the calculations.

        Inputs
        ------
        graph : np.array or array-like
            A two-dimensional square matrix of size [num_nodes, num_nodes] 
            containing either flags for connections (1 if an edge exists, 0 
            if not), or edge weights

        iterations : int
            Number of iterations of the algorithm to perform

        pos : np.array or array-like
             A one dimensional array containing a tuple or list of initial
             x and y coordinates

        sizes : np.array or array-like
            A one dimensional array containing node sizes, required for 
            adjust_sizes mode

        directed : bool
            Whether the supplied graph is directed or undirected

        barnes_hut_threta : float (default=1.2)
            The distance modifier parameter for barnes hut optimization, 
            if barnes_hut_optimize is False then this is ignored

        edge_weight_influence : float (default=0)
            The weighting factor on edges:
                - 0: No weighting applied
                - 1: Passed weights in graph variable used
                - Other: Passed weight ** (other) used
        
        gravity : float (default=0)
            A factor that controls how much attractive force any two nodes
            have on each other. The higher the value the smaller the graph

        jitter_tolerance : float (default=1)
            How reactive the graph is to small changes, the higher the value
            the less likely small changes are to affect the graph. Larger
            graphs may need a higher jitter tolerance

        scaling_ratio : float (default=2)
            How "spread out" the graph is. The higher the number the larger
            the graph

        adjust_sizes : bool
            Whether the use anti-collision mode. If True, must also pass
            sizes. Anti-collision mode prevents node overlap based on their 
            size
        
        barnes_hut_optimize : bool
            Whether to use regional optimization. For small graphs (n<500)
            this will slow down the calculation. As the graph gets lasrger
            the more speed gains to capture from using theis mode.

        lin_log_mode : bool
            Whether to use log based distance. When active, neighborhoods
            tend to be tighter, leaving more white space between neighborhoods

        outbound_attraction_distribution : bool
            Wehther to use the dissuade hubs option. When active, nodes with
            high indegree are more central while nodes with high outdegree 
            are pushed to the periphery.

        strong_gravity_mode : bool
            Whether to use the strong gravity mode. When active, nodes will
            have a much higher gravity, creating a much more compact graph.
        """
        
        self.graph = graph
        self.iterations = iterations
        self.pos = pos
        self.sizes = sizes
        
        #Barnes Hut
        self.barnes_hut_theta = barnes_hut_theta
        self.barnes_hut_optimize = barnes_hut_optimize
        
        #Factors
        self.gravity = gravity
        self.scaling_ratio = scaling_ratio
        self.jitter_tolerance = jitter_tolerance
        self.edge_weight_influence = edge_weight_influence
        
        #Flags
        self.adjust_sizes = adjust_sizes
        self.lin_log_mode = lin_log_mode
        self.strong_gravity_mode = strong_gravity_mode
        self.outbound_attraction_distribution = outbound_attraction_distribution
        
        #Speed parameters
        self.speed = 1
        self.speed_efficiency = 1
        
        #Queue
        self.queue = Queue()
        
        #Initialize layout data
        nodes = []
        for i in range(len(graph)):
            node = {'id': 0, 'x': 0, 'y': 0, 'dx' : 0, 'dy' : 0, 'old_dx' : 0, 
                    'old_dy': 0, 'mass' : 0, 'size' : None}
            
            node['id'] = i
            if pos:
                node['x'] = pos[i][0]
                node['y'] = pos[i][1]
            else:
                node['x'] = random.random()
                node['y'] = random.random()
            
            if sizes:
                node['size'] = sizes[i]

            node['mass'] = 1 + len(np.where(graph[i] > 0)[0])
            nodes.append(node)
            
        self.nodes = nodes
            
        edges = []
        if directed:
            x_cor, y_cor = graph.nonzero()
        else:
            upper_triangle = np.triu(graph)
            x_cor, y_cor = upper_triangle.nonzero()
        
        upper_triangle = np.triu(graph)
        x_cor, y_cor = upper_triangle.nonzero()
            
        for i in range(len(x_cor)):
            edge = {'source' : 0, 'target' : 0, 'weight' : 0}
            edge['source'] = x_cor[i]
            edge['target'] = y_cor[i]
            edge['weight'] = graph[x_cor[i], y_cor[i]]
            edges.append(edge)
            
        self.edges = edges
    
    def run_algo(self):
        for i in range(self.iterations):
            self.go_algo()
        return [(n['x'], n['y']) for n in self.nodes]
    
    def go_algo(self):
        
        #Initialize layout data
        for node in self.nodes:
            node['old_dx'] = node['dx']
            node['old_dy'] = node['dy']
            node['dx'] = 0
            node['dy'] = 0
        
        #If Barnes Hut active, initialize root region
        if self.barnes_hut_optimize:
            root_region = self.update_mass_and_geometry(self.nodes)
            root_region = self.build_subregions(root_region)
        
        #If outbountAttractionDistribution active, compensate
        if self.outbound_attraction_distribution:
            self.outbound_att_compensation = np.mean([n['mass'] for n in self.nodes])
        else:
            self.outbound_att_compensation = 1
        
        #buildRepulsion
        if self.barnes_hut_optimize:
            for i in range(len(self.nodes)):
                self.queue.put([i, root_region])
                target_thread = self.repulsion_region_thread
                
        else:
            for i in range(len(self.nodes)):
                for j in range(i):
                    self.queue.put([i,j])
                    target_thread = self.repulsion_thread
        
        #Multi-threaded
        num_cores = multiprocessing.cpu_count()
        for i in range(num_cores):
            repulse = Thread(target=target_thread)
            repulse.daemon = True
            repulse.start()
                
        #buildGravity
        for n in self.nodes:
            self.gravity_force(n, self.gravity, self.scaling_ratio, self.strong_gravity_mode)
        
        #buildAttraction
        for e in self.edges:
            self.attraction_force(e, self.lin_log_mode, 
                                  self.outbound_attraction_distribution, 
                                  self.adjust_sizes)
        
        #Auto adjust speed
        total_swinging = 0
        total_effective_traction = 0
        for n in self.nodes:
            swinging = ((n['old_dx'] - n['dx'])**2 + (n['old_dy'] - n['dy'])**2)**(1/2)
            total_swinging += n['mass'] * swinging
            total_effective_traction += .5 * n['mass'] * ((n['old_dx'] + n['dx'])**2 + (n['old_dy'] + n['dy'])**2)**(1/2)
            
        #Optimize jitter tolerance
        #The 'right' jitter tolerance for this network. Bigger networks need more tolerance.
        #Denser networks need less tolerance. Totally empiric.
        estimated_optimal_jitter_tolerance = .05 * len(self.nodes)**(1/2)
        min_jt = (estimated_optimal_jitter_tolerance)**(1/2)
        max_jt = 10
        jt = self.jitter_tolerance * max(min_jt, min(max_jt, (estimated_optimal_jitter_tolerance * total_effective_traction)/(len(self.nodes)**2)))
            
        min_speed_efficiency = .05

        #Protection against erratic behavior
        if (total_swinging / total_effective_traction) > 2:
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= .5
            jt = max(jt, self.jitter_tolerance)
            
        target_speed = (jt * self.speed_efficiency * total_effective_traction) / total_swinging
        
        #Speed efficiency is how the speed really corresponds to the swinging vs. convergence tradeoff
        #We adjust it slowly and carefully
        if total_swinging > (jt * total_effective_traction):
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= .7
        elif self.speed < 1000:
            self.speed_efficiency *= 1.3
            
        #But the speed shouldn't rise too quickly, since it would make the convergence drop dramatically
        max_rise = .5
        self.speed = self.speed + min(target_speed-self.speed, max_rise*self.speed)
        
        #Apply forces
        if (self.adjust_sizes) and (self.sizes):
            #If nodes overlap prevention is active, it's not possible to trust the swinging measure
            for n in self.nodes:
                
                #Adaptive auto-speed: the speed of each node is lowered
                #when the node swings
                swinging = n['mass'] * ((n['old_dx'] - n['dx'])**2 + (n['old_dy'] - n['dy'])**2)**(1/2)
                factor = (.1 * self.speed)/(1 + (self.speed * swinging)**(1/2))
                
                df = (n['dx']**2 + n['dy']**2)**(1/2)
                factor = min(factor*df, 10)/df
                
                x = n['x'] + (n['dx'] * factor)
                y = n['y'] + (n['dy'] * factor)
                
                n['x'] = x
                n['y'] = y
        
        else:
            for n in self.nodes:
                
                #Adaptive auto-speed: the speed of each node is lowered
                #when the node swings
                swinging = n['mass'] * ((n['old_dx'] - n['dx'])**2 + (n['old_dy'] - n['dy'])**2)**(1/2)
                factor = self.speed/(1 + (self.speed * swinging)**(1/2))
                
                x = n['x'] + (n['dx'] * factor)
                y = n['y'] + (n['dy'] * factor)
                
                n['x'] = x
                n['y'] = y
    
    def repulsion_thread(self):
        
        while self.queue.qsize:
            i,j = self.queue.get()
            self.repulsion_force(i, j, self.adjust_sizes, self.scaling_ratio)
            self.queue.task_done()
    
    def repulsion_region_thread(self):
        
        while self.queue.qsize:
            i,r = self.queue.get()
            
            if len(self.nodes) < 2:
                self.repulsion_force_region(0, 1, self.adjust_sizes, self.scaling_ratio)
                self.queue.task_done()
                
            else:
                n = self.nodes[i]
                distance = ((n['x'] - r['mass_center_x'])**2 + (n['y'] - r['mass_center_y'])**2) ** (1/2)
                
                if distance * self.barnes_hut_theta > r['size']:
                    self.repulsion_force_region(i, r, self.adjust_sizes, self.scaling_ratio)
                    
                else:
                    for subregion in r['subregions']:
                        self.repulsion_force_region(i, subregion, self.adjust_sizes, self.scaling_ratio)
                
                self.queue.task_done()
            
    def repulsion_force(self, i, j, adjust_by_size, coefficient):
        
        n1 = self.nodes[i]
        n2 = self.nodes[j]
            
        #Get the distance
        x_dist = n1['x'] - n2['x']
        y_dist = n1['y'] - n2['y']

        if (adjust_by_size) and (self.sizes):

            #linRepulsion_antiCollision
            distance = (x_dist**2 + y_dist**2)**(1/2) - n1['size'] - n2['size']

            if distance > 0:
                #Gephi code has double division : http://bit.ly/29CNteT
                #But paper does not: http://bit.ly/29DRQwe
                #factor = (coefficient * n1['mass'] * n2['mass'])/distance
                factor = coefficient * n1['mass'] * n2['mass'] / distance / distance

            elif distance < 0:
                factor = 100 * coefficient * n1['mass'] * n2['mass']

        else:

            #linRepulsion
            distance = (x_dist**2 + y_dist**2)**(1/2)
            if distance > 0:
                #Gephi code has double division : http://bit.ly/29CNteT
                #But paper does not: http://bit.ly/29DRQwe
                #factor = coefficient * n1['mass'] * n2['mass'] / distance
                factor = coefficient * n1['mass'] * n2['mass'] / distance / distance
                
        n1['dx'] += x_dist * factor
        n1['dy'] += y_dist * factor

        n2['dx'] -= x_dist * factor
        n2['dy'] -= y_dist * factor

        self.nodes[i] = n1
        self.nodes[j] = n2
                
    def repulsion_force_region(self, i, r, adjust_by_size, coefficient):
        
        n = self.nodes[i]
            
        #Get the distance
        x_dist = n['x'] - r['mass_center_x']
        y_dist = n['y'] - r['mass_center_y']
        distance = (x_dist**2 + y_dist**2)**(1/2)

        if (adjust_by_size) and (self.sizes):

            #linRepulsion_antiCollision
            if distance > 0:
                factor = coefficient * n['mass'] * r['mass'] / distance / distance

            elif distance < 0:
                factor = -coefficient * n1['mass'] * r['mass'] / distance

        else:

            #linRepulsion
            if distance > 0:
                factor = coefficient * n['mass'] * r['mass']/distance/distance

        n['dx'] += x_dist * factor
        n['dy'] += y_dist * factor

        self.nodes[i] = n
                
    def gravity_force(self, n, gravity, scaling_ratio, strong_gravity):
        
        x_dist = n['x']
        y_dist = n['y']
        distance = (x_dist**2 + y_dist**2)**(1/2)

        #strongGravity
        if strong_gravity:
            factor = scaling_ratio * n['mass'] * (gravity/scaling_ratio)
            n['dx'] -= x_dist * factor
            n['dy'] -= y_dist * factor

        else:
            factor = scaling_ratio * n['mass'] * (gravity/scaling_ratio) / distance
            n['dx'] -= x_dist * factor
            n['dy'] -= y_dist * factor
                    
    def attraction_force(self, e, log_attraction, distributed_attraction, adjust_by_size):
        
        n1 = self.nodes[e['source']]
        n2 = self.nodes[e['target']]

        if self.edge_weight_influence == 0:
            weight = 1
        elif self.edge_weight_influence == 1:
            weight = e['weight']
        else:
            weight = e['weight'] ** self.edge_weight_influence
            
        x_dist = n1['x'] - n2['x']
        y_dist = n1['y'] - n2['y']

        if (adjust_by_size) and (self.sizes):
            distance = (x_dist**2 + y_dist**2)**(1/2) - n1['size'] - n2['size']

            if log_attraction:
                if distributed_attraction:
                    #logAttraction_degreeDistributed_antiCollision
                    factor = -self.outbound_att_compensation * weight * np.log(1+distance) / distance / n1['mass']

                else:
                    #logAttraction_antiCollision
                    factor = -self.outbound_att_compensation * weight * np.log(1+distance) / distance

            else:
                if distributed_attraction:
                    #linAttraction_degreeDistributed_antiCollision
                    factor = -self.outbound_att_compensation * weight / n1['mass']

                else:
                    #linAttraction_antiCollision
                    factor = -self.outbound_att_compensation * weight

        else:
            distance = (x_dist**2 + y_dist**2)**(1/2)

            if log_attraction:
                if distributed_attraction:
                    #logAttraction_degreeDistributed
                    factor = -self.outbound_att_compensation * weight * np.log(1+distance) / distance / n1['mass']

                else:
                    #logAttraction
                    factor = -self.outbound_att_compensation * weight * (np.log(1+distance)) / distance

            else:
                if distributed_attraction:
                    #linAttraction_massDistributed
                    factor = -self.outbound_att_compensation * weight / n1['mass']

                else:
                    #linAttraction
                    factor = -self.outbound_att_compensation * weight

        if distance > 0:
            n1['dx'] += x_dist * factor
            n1['dy'] += y_dist * factor

            n2['dx'] -= x_dist * factor
            n2['dy'] -= y_dist * factor

            self.nodes[e['source']] = n1
            self.nodes[e['target']] = n2
            
    def update_mass_and_geometry(self, nodes):
        r = {'mass': 0, 'mass_center_x': 0, 'mass_center_y': 0, 'size': 0, 'nodes': [], 'subregions': []}
        r['nodes'] = nodes
        
        if len(nodes) > 1:
            mass = 0
            mass_sum_x = 0
            mass_sum_y = 0
            
            for n in nodes:
                mass += n['mass']
                mass_sum_x += n['x'] * n['mass']
                mass_sum_y += n['y'] * n['mass']
            
            r['mass'] = mass
            r['mass_center_x'] = mass_sum_x / mass
            r['mass_center_y'] = mass_sum_y / mass
            
            size = 0
            for n in nodes:
                distance = ((n['x'] - r['mass_center_x'])**2 + (n['y'] - r['mass_center_y'])**2)**(1/2)
                size = max(size, 2 * distance)
            r['size'] = size
            
        return r
        
    def build_subregions(self, r):
        
        if len(r['nodes']) > 1:
            left_nodes = []
            right_nodes = []
            
            for n in r['nodes']:
                if n['x'] < r['mass_center_x']:
                    left_nodes.append(n)
                else:
                    right_nodes.append(n)
                    
            top_left_nodes = []
            bottom_left_nodes = []
            for n in left_nodes:
                if n['y'] < r['mass_center_y']:
                    top_left_nodes.append(n)
                else:
                    bottom_left_nodes.append(n)
                    
            top_right_nodes = []
            bottom_right_nodes = []
            for n in right_nodes:
                if n['y'] < r['mass_center_y']:
                    top_right_nodes.append(n)
                else:
                    bottom_right_nodes.append(n)
                    
            if len(top_left_nodes) > 0:
                if len(top_left_nodes) < len(self.nodes):
                    subregion = self.update_mass_and_geometry(top_left_nodes)
                    r['subregions'].append(subregion)
                else:
                    for n in top_left_nodes:
                        subregion = self.update_mass_and_geometry([n])
                        r['subregions'].append(subregion)
                        
            if len(bottom_left_nodes) > 0:
                if len(bottom_left_nodes) < len(self.nodes):
                    subregion = self.update_mass_and_geometry(bottom_left_nodes)
                    r['subregions'].append(subregion)
                else:
                    for n in bottom_left_nodes:
                        subregion = self.update_mass_and_geometry([n])
                        r['subregions'].append(subregion)
                        
            if len(top_right_nodes) > 0:
                if len(top_right_nodes) < len(self.nodes):
                    subregion = self.update_mass_and_geometry(top_right_nodes)
                    r['subregions'].append(subregion)
                else:
                    for n in top_right_nodes:
                        subregion = self.update_mass_and_geometry([n])
                        r['subregions'].append(subregion)
                        
            if len(bottom_right_nodes) > 0:
                if len(bottom_right_nodes) < len(self.nodes):
                    subregion = self.update_mass_and_geometry(bottom_right_nodes)
                    r['subregions'].append(subregion)
                else:
                    for n in bottom_right_nodes:
                        subregion = self.update_mass_and_geometry([n])
                        r['subregions'].append(subregion)
            
            for i in range(len(r['subregions'])):
                r['subregions'][i] = self.build_subregions(r['subregions'][i])
                
            return r
