# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:32:24 2018

@author: jackr
"""

#Data Structure stores collection of Superpixels as vertices
import numpy as np
class Graph:
    
    def __init__(self):
        self.vert_dict = {} #vertex dictionary
        self.num_vertices = 0
        
    def __iter__(self):
        return iter(self.vert_dict.values())
    
    def add_vertex(self, node):
        self.num_vertices += 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex
    
    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None
    
    def get_vertices(self):
        return self.vert_dict.keys()
    
    def add_vertex_from_segments(self):
        verts = self.segments_unique
        
        for i in range(len(self.segments_unique)):
            self.add_vertex(verts[i])
            if(self.get_vertex(i) == 0):
                centre = 0,0
            else:
                centre = self.regions[i-1].centroid
            self.get_vertex(i).add_centroid(centre)
    
    def add_edge(self, frm, to, cost=0.0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
            
        if to not in self.vert_dict:
            self.add_vertex(to)
            
        self.vert_dict[frm].add_neighbour(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbour(self.vert_dict[frm], cost)

#Data Structure for individual Superpixel

import numpy as np

class Vertex:
    
    def __init__(self, node):
        self.id = node
        self.adjacent = {} # list of connected vertices
        self.adjacent_id = [] #list of adjacent ids
        self.lab = np.zeros(3)
        self.hasPath = False
        self.cx = 0
        self.cy = 0
        
    def __str__(self):
        return str(self.id) + ' adjacent: ' +str([x.id for x in self.adjacent])+ '  no: ' + str(len(self.adjacent))
    
    def add_neighbour(self, neighbour, weight=0.0):
        self.adjacent[neighbour] = weight
        self.adjacent_id.append(neighbour.get_id())
        
    def get_connections(self):
        return self.adjacent.keys() #returns available keys
    
    def get_id(self):
        return self.id
    
    def get_weight(self, neighbour):
        return self.adjacent[neighbour]

    def set_inputs(input_X):
        self.input_X = input_X
        