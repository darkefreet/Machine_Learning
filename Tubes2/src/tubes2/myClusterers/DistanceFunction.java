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
public interface DistanceFunction {
    double distanceOf(Instance a, Instance b) throws Exception;
}
