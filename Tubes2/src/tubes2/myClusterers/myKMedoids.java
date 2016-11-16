/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2.myClusterers;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
 */
public class myKMedoids extends weka.clusterers.AbstractClusterer{
    
    int k;
    DistanceFunction distanceFunction;
    public myKMedoids(int k, DistanceFunction distanceFunction){
        this.k = k;
        this.distanceFunction = distanceFunction;
    }
    public myKMedoids(int k){
        this.k = k;
        this.distanceFunction = new EuclideanDistance();
    }
    
    @Override
    public int clusterInstance(Instance instance)
                    throws java.lang.Exception{
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void buildClusterer(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }
    
}
