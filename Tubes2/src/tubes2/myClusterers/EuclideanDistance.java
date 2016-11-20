/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2.myClusterers;

import weka.core.Instance;

/**
 *
 * @author nim_13512501
 */
public class EuclideanDistance implements DistanceFunction{

    @Override
    public double distanceOf(Instance a, Instance b) throws Exception{
        if (a.numAttributes() != b.numAttributes())
            throw new Exception("number of attributes doesn't match");
        double sumSquares = 0;
        for (int i=0; i<a.numAttributes(); i++){
            double Error = a.value(i)-b.value(i);
            sumSquares+= Error*Error;
        }
        return Math.sqrt(sumSquares);
    }
    
}
