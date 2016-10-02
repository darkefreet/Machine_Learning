/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Helper;

import weka.core.Instance;
import weka.classifiers.Classifier;

/**
 *
 * @author ivan
 */
public class ClassifyHelper  {

    public static void clasifyInstance(Classifier cls, Instance inst) throws Exception {
        double result = cls.classifyInstance(inst);
        inst.setClassValue(result);
    }

}
