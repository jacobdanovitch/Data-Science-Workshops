#############################################################################
# Full Imports

import math
import random
import plotly
from plotly.graph_objs import Scatter, Scatter3d, Layout

def main():

    num_points = 20
    dimensions = 2
    lower = 0
    upper = 200

    points = [
        makeRandomPoint(dimensions, lower, upper) for i in range(num_points)
    ]

    num_clusters = 3
    cutoff = 0.2


    clusters = kmeans(points, num_clusters, cutoff)
    for i, c in enumerate(clusters):
        for p in c.points:
            print(" Cluster: ", i, "\t Point :", p)

    if dimensions in [2, 3] and plotly:
        print("Plotting points, launching browser ...")
        plotClusters(clusters, dimensions)

class Point(object):
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster(object):
    def __init__(self, points):
        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        self.points = points

        self.n = points[0].n

        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)

def kmeans(points, k, cutoff):

    initial = random.sample(points, k)
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            # After finding the cluster the smallest distance away
            # set the point to belong to that cluster
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print("Converged after %s iterations" % loopCounter)
            break
    return clusters

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ERROR: non comparable points")

    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        accumulatedDifference += squareDifference
    distance = math.sqrt(accumulatedDifference)

    return distance

def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for _ in range(n)])
    return p

def plotClusters(data, dimensions):
    '''
    This uses the plotly offline mode to create a local HTML file.
    This should open your default web browser.
    '''
    if dimensions not in [2, 3]:
        raise Exception("Plots are only available for 2 and 3 dimensional data")

    # Convert data into plotly format.
    traceList = []
    for i, c in enumerate(data):
        # Get a list of x,y coordinates for the points in this cluster.
        cluster_data = []
        for point in c.points:
            cluster_data.append(point.coords)

        trace = {}
        centroid = {}
        if dimensions == 2:
            # Convert our list of x,y's into an x list and a y list.
            trace['x'], trace['y'] = zip(*cluster_data)
            trace['mode'] = 'markers'
            trace['marker'] = {}
            trace['marker']['symbol'] = i
            trace['marker']['size'] = 12
            trace['name'] = "Cluster " + str(i)
            traceList.append(Scatter(**trace))
            # Centroid (A trace of length 1)
            centroid['x'] = [c.centroid.coords[0]]
            centroid['y'] = [c.centroid.coords[1]]
            centroid['mode'] = 'markers'
            centroid['marker'] = {}
            centroid['marker']['symbol'] = i
            centroid['marker']['color'] = 'rgb(200,10,10)'
            centroid['name'] = "Centroid " + str(i)
            traceList.append(Scatter(**centroid))
        else:
            symbols = [
                "circle",
                "square",
                "diamond",
                "circle-open",
                "square-open",
                "diamond-open",
                "cross", "x"
            ]
            symbol_count = len(symbols)
            if i > symbol_count:
                print("Warning: Not enough marker symbols to go around")
            # Convert our list of x,y,z's separate lists.
            trace['x'], trace['y'], trace['z'] = zip(*cluster_data)
            trace['mode'] = 'markers'
            trace['marker'] = {}
            trace['marker']['symbol'] = symbols[i]
            trace['marker']['size'] = 12
            trace['name'] = "Cluster " + str(i)
            traceList.append(Scatter3d(**trace))
            # Centroid (A trace of length 1)
            centroid['x'] = [c.centroid.coords[0]]
            centroid['y'] = [c.centroid.coords[1]]
            centroid['z'] = [c.centroid.coords[2]]
            centroid['mode'] = 'markers'
            centroid['marker'] = {}
            centroid['marker']['symbol'] = symbols[i]
            centroid['marker']['color'] = 'rgb(200,10,10)'
            centroid['name'] = "Centroid " + str(i)
            traceList.append(Scatter3d(**centroid))

    title = "K-means clustering with %s clusters" % str(len(data))
    plotly.offline.plot({
        "data": traceList,
        "layout": Layout(title=title)
    })

if __name__ == "__main__":
    main()
