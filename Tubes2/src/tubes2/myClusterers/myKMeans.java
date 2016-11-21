/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2.myClusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
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
    
    Instances template;
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
        centroids = new Instance[k];
        
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
    
    /**
     * initializes centroids with the longest distances
     * @param i 
     */
    public void initializeCentroidsDistance(Instances i) throws Exception{
        int n = i.numInstances();
        if (n<k)
            throw new Exception("n<k");
        centroids = new Instance[k];

        //CARI RATA-RATA DARI K-1 DISTANCE TERTINGGI MASING-MASING SIMPUL
        Double meanN[] = new Double[n];
        Double allDistances[][] = new Double[n][n];
        for(int a = 0; a<n;a++){
            Double arr[] = new Double[n];
            for(int b = 0; b<n; b++){
                allDistances[a][b] = distanceFunction.distanceOf(i.instance(a),i.instance(b));
                arr[b] = allDistances[a][b];
            }
            Arrays.sort(arr);
            Double total = 0.0;
            for(int c = 0; c<k-1;c++){
                total+=arr[n-1-c];
            }
            meanN[a] = total / (k-1);
        }
        
        //AMBIL RATA-RATA TERBESAR
        Double max = meanN[0];
        int indexMax = 0;
        for(int a=1;a<n;a++){
//            System.out.println(a + " bernilai "+ meanN[a]);
            if(meanN[a]>max){
                indexMax = a;
                max = meanN[a];
            }
        }
        System.out.println("index max " +indexMax);
        
        //AMBIL K-1 TITIK DARI SIMPUL DENGAN RATAAN JARAK TERBESAR
        ArrayList<Integer> centroidPos = new ArrayList<Integer>();
        Double temp[] = new Double[n];
        temp = Arrays.copyOf(allDistances[indexMax],n);
        Arrays.sort(temp);
        centroidPos.add(indexMax);
        for(int a=0;a<k-1;a++){
            for(int b=0;b<n;b++){
                if(temp[n-1-a]==allDistances[indexMax][b]){
                    centroidPos.add(b);
                    //so the element wont be picked again
                    allDistances[indexMax][b] = -1.0;
                    break;
                }
            }
        }
        for (int a=0;a<k;a++){
            centroids[a] = i.instance(centroidPos.get(a));
        }
    }
    
    Instances [] oldClusters;
    public void assignClusters(Instances instances) throws Exception {
        oldClusters = clusters;
        clusters = new Instances[k];
        for (int i=0;i<k;i++) clusters[i]=new Instances(instances, instances.numAttributes());
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
    
    public boolean clusterChanged(){
        if (oldClusters==null) return true;
        for (int i=0;i<k;i++){
            if (clusterDifferent(oldClusters[i],clusters[i])){
                return true;
            }
        }
        return false;
    }
    
    public boolean clusterDifferent(Instances a, Instances b){
        for (int i=0;i<a.numInstances();i++){
            if (instanceDifferent(a.instance(i),b.instance(i)))
                return true;
        }
        return false;
    }
    
    public boolean instanceDifferent(Instance a, Instance b){
        for (int i=0;i<a.numAttributes();i++){
            if (a.value(i)!=b.value(i))
                return true;
        }
        return false;
    }
    
    public void moveCentroids() throws Exception{
        for (int i=0;i<k;i++){
            centroids[i]=mean(clusters[i]);
        }
    }
    
    public Instance mean(Instances i){
        Instance mean = new weka.core.Instance(i.numAttributes());
        for (int j=0;j<i.numAttributes();j++){
            double meanValue = meanValue(i,j);
            mean.setValue(j, meanValue);
        }
        return mean;
    }
    
    public double meanValue(Instances i, int attrIndex){
        double sum = 0;
        for (int j=0;j<i.numInstances();j++){
            sum+=i.instance(j).value(attrIndex);
        }
        return sum/i.numInstances();
    }

    int iter;
    @Override
    public void buildClusterer(Instances i) throws Exception {
        template = new Instances(i,0);
        //DENGAN RANDOM
        initializeCentroids(i);
        //TANPA RANDOM
//        initializeCentroidsDistance(i);
        assignClusters(i);
        iter=0;
        do {
            moveCentroids();
            assignClusters(i);
            iter++;
        }while(clusterChanged());
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    public String toString(){
        try {
            return "myKMeans\n"
                    + "iterations:"+iter+"\n"
                    + "sum squared error: " + innerClusterSumSquaredError() + "\n"
                    + "centroids:\n"
                    + centroidsToString() +"\n";
        } catch (Exception ex) {
            Logger.getLogger(myKMeans.class.getName()).log(Level.SEVERE, null, ex);
            return ex.toString();
        }
    }
    
    public String centroidsToString(){
        Instances centroidInstances = new Instances(template,0);
        for (int i=0;i<k;i++){
            centroidInstances.add(centroids[i]);
        }
        return centroidInstances.toString();
    }
    
    public String clustersToString() throws Exception{
        String retval = "";
        for (int i=0;i<k;i++){
            retval+= "cluster-"+i;
            retval+=clusters[i];
        }
        return retval;
    }
    
    public double innerClusterSumSquaredError() throws Exception{
        double sumError = 0;
        for (int clusterIndex=0;clusterIndex<k;clusterIndex++){
            Instances clusterInstances = clusters[clusterIndex];
            Instance centroid = centroids[clusterIndex];
            for (int i=0;i<clusterInstances.numInstances();i++){
                Instance clusterInstance = clusterInstances.instance(i);
                double dist = distanceFunction.distanceOf(clusterInstance, centroid);
                sumError+=dist*dist;
            }
        }
        return Math.sqrt(sumError);
    }
}
