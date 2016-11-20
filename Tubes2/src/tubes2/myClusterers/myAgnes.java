/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2.myClusterers;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;

/**
 *
 * @author ivan
 */
public class myAgnes extends weka.clusterers.AbstractClusterer {
    
    int k;
    int iter;
    Instances template;
    Instances [] clusters;
    
    DistanceFunction distanceFunction;
    public myAgnes(int k){
        this.k = k;
        this.distanceFunction = new EuclideanDistance();
    }
    
    @Override
    public void buildClusterer(Instances i) throws Exception {
        template = new Instances(i, 0);
        initializeClusters(i);
        iter=0;
        do {
            joinCluster();
            iter++;
        } while (clusters.length > k);
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }
    
    public String toString(){
        try {
            return "myAgnes\n"
                    + "iterations:" + iter + "\n"
                    + "centroids:\n"
                    + printClusters() + "\n";
        } catch (Exception ex) {
            Logger.getLogger(myKMeans.class.getName()).log(Level.SEVERE, null, ex);
            return ex.toString();
        }
    }
    
    public String printClusters() {
        String str = "";
        for (int i = 0; i < clusters.length; i++) {
            str += "Cluster-" + i + "\n";
            Instances instances = clusters[i];
            for (int j = 0; j < instances.numInstances(); j++) {
                str += instances.instance(j) + "\n";
            }
        }
        return str;
    }
    
    public void initializeClusters(Instances data) {
        int numInstances = data.numInstances();
        clusters = new Instances[numInstances];
        for (int i = 0; i < numInstances; i++) {
            clusters[i] = new Instances(data, 0);
            clusters[i].add(data.instance(i));
        }
    }
    
    public void joinCluster() throws Exception {
        int numCluster = clusters.length;
        double[][] clusterDistance = getClusterDistance();
        
        int nearestDistanceI = -1;
        int nearestDistanceJ = -1;
        double nearestDistance = clusterDistance[0][1];
        
        for (int i = 0; i < numCluster; i++) {
            for (int j = i + 1; j < clusters.length; j++) {
                double distance = clusterDistance[i][j];
                if (distance < nearestDistance) {
                    nearestDistance = clusterDistance[i][j];
                    nearestDistanceI = i;
                    nearestDistanceJ = j;
                }
            }
        }
        
        Instances[] newClusters = new Instances[clusters.length - 1];
        newClusters[0] = new Instances(clusters[nearestDistanceI], 0);
        for (int i = 0; i < clusters[nearestDistanceI].numInstances(); i++) {
            newClusters[0].add(clusters[nearestDistanceI].instance(i));
        }
        for (int i = 0; i < clusters[nearestDistanceJ].numInstances(); i++) {
            newClusters[0].add(clusters[nearestDistanceJ].instance(i));
        }

        int newClusterIdx = 1;
        for (int i = 0; i < clusters.length; i++) {
            if ((i != nearestDistanceI) && (i != nearestDistanceJ)) {
                newClusters[newClusterIdx] = clusters[i];
                newClusterIdx++;
            }
        }
        clusters = newClusters;
    }
    
    public double[][] getClusterDistance() throws Exception {
        int numCluster = clusters.length;
        double[][] distances = new double[numCluster][numCluster];
        
        for (int i = 0; i < numCluster; i++) {
            for (int j = i + 1; j < numCluster; j++) {
                distances[i][j] = calculateClusterDistance(clusters[i], clusters[j]);
            }
        }
        
        return distances;
    }
    
    public double calculateClusterDistance(Instances a, Instances b) throws Exception {
        double[][] distances = new double[a.numInstances()][b.numInstances()];
        for (int i = 0; i < a.numInstances(); i++) {
            for (int j = 0; j < b.numInstances(); j++) {
                distances[i][j] = distanceFunction.distanceOf(a.instance(i), b.instance(j));
            }
        }
        
        double maxDistance = distances[0][0];
        for (int i = 0; i < a.numInstances(); i++) {
            for (int j = 0; j < b.numInstances(); j++) {
                double distance = distances[i][j];
                if (distance > maxDistance) {
                    maxDistance = distance;
                }
            }
        }

        return maxDistance;
    }
    

}
