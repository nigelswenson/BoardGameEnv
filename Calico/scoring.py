import numpy as np
from gridTypes import Tile
class BasicClusterManager:
    # This handles clusters of size 3. serves as a template for cluster scoring 
    def __init__(self, cluster_size = 3):
        self.parent = {}
        self.values = {}
        self.cluster_size = {}
        self.clusters = set()
        self.cluster_point_threshold = cluster_size

    def find(self, val):
        if self.parent[val] == val:
            return val
        else:
            return self.find(self.parent[val])
        
    def union(self, new, old):
        size_new = self.cluster_size[new]
        size_old = self.cluster_size[old]
        if (size_new <= self.cluster_point_threshold) | (size_old <= self.cluster_point_threshold-1) and (self.parent[new] != self.parent[old]):
            # merge new into old
            self.parent[new] = self.parent[old]
            self.cluster_size[old] += self.cluster_size[new]
            self.cluster_size[new] = 0
            self.clusters.remove(new)
    
    def addTile(self, value, location, neighbors):
        self.parent[location] = location
        self.cluster_size[location] = 1
        self.values[location] = value
        self.clusters.add(location)
        for neighbor in neighbors:
            if neighbor[0] in self.parent:
                if self.values[neighbor[0]] == value:
                    # there is a match, try to do a union
                    self.union(self.find(location), self.find(neighbor[0]))
    
    def getPointClusters(self):
        big_clusters = []
        for location in self.clusters:
            if self.cluster_size[location] >= self.cluster_point_threshold:
                big_clusters.append((location,self.values[location], self.cluster_size[location]))
        return big_clusters

class CatClusterManager(BasicClusterManager):
    def __init__(self, cluster_size=3, pattern_ids_used=[], catValue=None):
        super().__init__(cluster_size)
        self.ids = pattern_ids_used
        self.catValue = catValue

    def addTile(self, value, location, neighbors):
        if value in self.ids:
            return super().addTile(value, location, neighbors)

class LineClusterManager(BasicClusterManager):
    pass    

class PointTile:
    def __init__(self, points):
        self.single_points = points[0]
        self.double_points = points[1]
        self.finished = False
        self.score = 0

    def calculate_score(self,adjacentTiles):
        pass

    def get_score(self):
        return self.score
    
class AAABBBTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            _, color_counts = np.unique(colors, return_counts=True)
            pattern_fit = (len(pattern_counts) == 2) and (pattern_counts[0]==3) and (pattern_counts[1]==3)
            color_fit = (len(color_counts) == 2) and(color_counts[0]==3) and (color_counts[1]==3)
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True

class AAAABBTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            pattern_counts.sort()
            _, color_counts = np.unique(colors, return_counts=True)
            color_counts.sort()
            pattern_fit = (len(pattern_counts) == 2) and (pattern_counts[0]==2) and (pattern_counts[1]==4)
            color_fit = (len(color_counts) == 2) and(color_counts[0]==2) and (color_counts[1]==4)
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True

class AABBCCTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            _, color_counts = np.unique(colors, return_counts=True)
            pattern_fit = (len(pattern_counts) == 3) and (pattern_counts[0]==2) and (pattern_counts[1]==2) and (pattern_counts[2]==2)
            color_fit =(len(color_counts) == 3) and (color_counts[0]==2) and (color_counts[1]==2) and (color_counts[2]==2)
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True

class AABBCDTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            _, color_counts = np.unique(colors, return_counts=True)
            pattern_fit = (len(pattern_counts) == 4) and (pattern_counts[2]==2) and (pattern_counts[3]==2)
            color_fit =(len(color_counts) == 4) and (color_counts[2]==2) and (color_counts[3]==2)
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True

class AAABBCTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            _, color_counts = np.unique(colors, return_counts=True)
            pattern_fit = (len(pattern_counts) == 3) and (pattern_counts[0]==1) and (pattern_counts[1]==2) and (pattern_counts[2]==3)
            color_fit =(len(color_counts) == 3) and (color_counts[0]==1) and (color_counts[1]==2) and (color_counts[2]==3)
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True

class NotEqualTile(PointTile):
    def calculate_score(self, adjacentTiles):
        if not self.finished:
            patterns =[]
            colors =[]
            for _, tile in adjacentTiles:
                if type(tile) is Tile:
                    patterns.append(tile.pattern)
                    colors.append(tile.color)
                else:
                    return
            _, pattern_counts = np.unique(patterns, return_counts=True)
            _, color_counts = np.unique(colors, return_counts=True)
            pattern_fit = len(pattern_counts) == 6
            color_fit = len(color_counts) == 6
            if pattern_fit and color_fit:
                self.score = self.double_points
            elif pattern_fit ^ color_fit:
                self.score = self.single_points
            else:
                self.score = 0
            self.finished = True
'''
# series of pattern things
def find3Color(board, color, existing_groups):
    # current implementation does not join two separate groups when they are linked.
    # we want this behavior when there are already pins down but dont want this 
    # when there are not pins associated with them. 
    for tile in board:
        if type(tile[1]) != Tile:
            continue
        # is it the right color?
        if (tile[1].color == color):
            in_groups = [e.contains(tile) for e in existing_groups]
            # is it already in a group?
            if not any(in_groups):
                neighbors = tile.getAdjacent(tile[0])
                # is it ajacent to a group? If so add it to that group
                done = checkandAddToExistingGroups(neighbors,existing_groups,tile)
                if not done:
                    # since it isnt, make a new group and add that to the list
                    existing_groups.append(TileGroup([tile]))
    return existing_groups

def checkandAddToExistingGroups(neighbors,existing_groups,tile):
    for n in neighbors:
        for e in existing_groups:
            if e.contains(n):
                # its neighbor is in a group. add this tile to that group
                e.append(tile)
                return True
    return False

    # group is a list of tiles that share some ajacency. We want to search them and build a new list of tiles ajacent to all of them
    # and do it again if we find a new one. 
    # thought, iteratively check the ajacent tiles that have the desired feature
    # if they have a neighbor, add that to a list of things to check
    # then check that list again to build chains
'''