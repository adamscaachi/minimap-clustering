import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, MeanShift

def get_icon_positions(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    positions = []
    for i, centroid in enumerate(centroids[1:], start=1): 
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 < area < 2000:
            positions.append(centroid - 3)
    return positions

def _initialise_clustering(method, params):
    if method == 'affinity':
        return AffinityPropagation(**params)
    elif method == 'agglomerative':
        return AgglomerativeClustering(**params)
    elif method == 'birch':
        return Birch(**params)
    elif method == 'dbscan':
        return DBSCAN(**params)
    elif method == 'meanshift':
        return MeanShift(**params)

def get_clusters(X, method, params = {}):
    clustering = _initialise_clustering(method, params)
    clusters = clustering.fit_predict(X)
    labels = np.unique(clusters)
    cluster_points = []  
    for label in labels:
        if label == -1:
            continue 
        points = [X[i] for i in range(len(X)) if clusters[i] == label]
        points = np.array(points) 
        cluster_points.append(points)
    return cluster_points

def plot_grid(image, cluster_points_list, algorithm_names):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    num_clusters = len(cluster_points_list)
    fig, axes = plt.subplots(1, num_clusters + 1, figsize = (5 * (num_clusters + 1), 7))
    ax = axes[0]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Raw Image', fontsize=20)
    for i, (cluster_points, algorithm_name) in enumerate(zip(cluster_points_list, algorithm_names), start=1):
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(algorithm_name, fontsize=20) 
        for points in cluster_points:
            ax.plot(points[:, 0], points[:, 1], '.', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=5)
            ax.plot(points[:, 0], points[:, 1], '.', markersize=25, alpha=1.0)
    plt.tight_layout()
    plt.savefig('clusters.png', dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    image = cv2.imread("images/1.png")
    X = get_icon_positions(image)
    affinity_clusters = get_clusters(X, 'affinity', {'damping': 0.5})
    agglomerative_clusters = get_clusters(X, 'agglomerative', {'n_clusters': None, 'distance_threshold': 800})
    birch_clusters = get_clusters(X, 'birch', {'threshold': 0.5, 'branching_factor': 50})
    dbscan_clusters = get_clusters(X, 'dbscan', {'eps': 140, 'min_samples': 2})
    meanshift_clusters = get_clusters(X, 'meanshift', {'bandwidth': 200})
    cluster_points_list = [affinity_clusters, agglomerative_clusters, birch_clusters, dbscan_clusters, meanshift_clusters]
    algorithm_names = ['Affinity Propagation', 'Agglomerative Clustering', 'BIRCH', 'DBSCAN', 'Mean Shift']
    plot_grid(image, cluster_points_list, algorithm_names)