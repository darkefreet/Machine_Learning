/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1.myClassifiers;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
 */
public class myID3 extends Classifier {

    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        
        
        
        
    }
    
    @Override
    public double classifyInstance(Instance instance){
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public double calculateAttributeProportion(Instances instances, Attribute attribute) {
        int numClasses = instances.classAttribute().numValues();
        
        return 0;
    }

    public myID3() {
    }
    
    public HashMap<String, Integer> getAttributeValues(Instances instances, Attribute attribute) {
        int numInstances = instances.numInstances();
        HashMap<String, Integer> values = new HashMap<String, Integer>();
        for (int i = 0; i < numInstances; i++) {
            String key = instances.instance(i).stringValue(attribute);
            if (values.containsKey(key)) {
                values.put(key, values.get(key) + 1);
            } else {
                values.put(key, 1);
            }
        }
        return values;
    }
    
    private Instances filterInstanceWithAttributeValue(Instances instances, Attribute attribute, String value) {
        Instances newInstances = new Instances(instances);
        newInstances.delete();
        int numInstances = instances.numInstances();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            if (instance.stringValue(attribute).equals(value)) {
                newInstances.add(instance);   
            }
        }
        return newInstances;
    }

    public double infoGain(Instances instances, Attribute attribute, double entropyOfSet){
        int numClasses = instances.classAttribute().numValues();
        int numInstances = instances.numInstances();
        HashMap<String, Integer> values = getAttributeValues(instances, attribute);
        double zigma = 0;
        Set<String> keys = values.keySet();
        for (int i = 0; i < keys.size(); i++) {
            String key = nthElement(keys, i);
            Instances instanceWithAttributeValue = filterInstanceWithAttributeValue(instances, attribute, key);
            zigma += (values.get(key) / numInstances) * entropy(instanceWithAttributeValue);
        }
        return entropyOfSet - zigma;
    }
    
    public double entropy(Instances instances){
        double result = 0;
        double[] proportion = calculateProportion(instances);
        for (double p : proportion){
            result -= Math.log(p)/Math.log(2)*p;
        }
        return result;
    }
    
    public double [] calculateProportion(Instances instances){
        int numClasses = instances.classAttribute().numValues();
        int numInstances = instances.numInstances();
        double [] result = new double[numClasses];
        int [] num = new int[numClasses];
        for (int i=0;i<numClasses;i++){
            num[i]=0;
        }
        for (int i=0;i<numInstances;i++){
            Instance instance = instances.instance(i);
            int classIndex = instance.classIndex();
            num[classIndex]++;
        }
        for (int i=0;i<numClasses;i++){
            result[i] = num[i]/numInstances;
        }
        return result;
    }
    
    public static final <T> T nthElement(Iterable<T> data, int n){
    int index = 0;
    for(T element : data){
        if(index == n){
            return element;
        }
        index++;
      }
    return null;
  }

}
