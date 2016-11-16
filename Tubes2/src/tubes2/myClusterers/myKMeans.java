/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2.myClusterers;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
 */
public class myKMeans extends weka.clusterers.AbstractClusterer {
    
    int k;
    DistanceFunction distanceFunction;
    public myKMeans(int k, DistanceFunction distanceFunction){
        this.k = k;
        this.distanceFunction = distanceFunction;
    }
    public myKMeans(int k){
        this.k = k;
        this.distanceFunction = new EuclideanDistance();
    }
    
    Instance [] centroids;
    Instances [] clusters;
    
    @Override
    public int clusterInstance(Instance instance)
                    throws java.lang.Exception{
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /**
     * initializes centroids randomly
     * @param i 
     */
    public void initializeCentroids(Instances i) throws Exception{
        int n = i.numInstances();
        if (n<k)
            throw new Exception("n<k");
        
        Set<Integer> centroidIndexSet = new HashSet<>();
        Random random = new Random();
        for (int j=0;j<k;j++){
            int centroidIndex;
            do{
                centroidIndex = random.nextInt(n);
            }while(centroidIndexSet.contains(centroidIndex));
            
            centroids[j]=i.instance(centroidIndex);
        }
    }
    
    public void assignClusters(Instances instances) throws Exception {
        clusters = new Instances[k];
        for (int i=0;i<instances.numInstances();i++){
            assignCluster(instances.instance(i));
        }
    }
    
    public void assignCluster(Instance instance) throws Exception{
        int clusterNum = nearestCentroid(instance);
        
        clusters[clusterNum].add(instance);
    }
    
    public int nearestCentroid(Instance instance) throws Exception{
        int iChosen = -1;
        double minDistance = Double.MAX_VALUE;
        for (int i=0;i<centroids.length;i++){
            double distance = distanceFunction.distanceOf(instance, centroids[i]);
            if (distance<minDistance){
                iChosen = i;
                minDistance = distance;
            }
        }
        return iChosen;
    }

    @Override
    public void buildClusterer(Instances i) throws Exception {
        initializeCentroids(i);
        assignClusters(i);
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }
    
}
