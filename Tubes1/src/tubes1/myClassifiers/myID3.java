/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1.myClassifiers;


import Helper.CustomFilter;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
 */
public class myID3 extends Classifier implements Serializable {
    
    private class TreeNode implements Serializable{
        public Attribute decision;
        public double branchValue;
        public int label;
        public List<TreeNode> branches;
        public TreeNode(){
            decision = null;
            label = -1;
            branchValue = -1;
            branches = new LinkedList<>();
        }
        public void addBranch(TreeNode node){
            branches.add(node);
        }
        public boolean isLeafNode(){
            return label >= 0;
        }
        public String stringify(int tabLevel){
            String retval = "";
            for (int i=0;i<tabLevel;i++){
                retval+='\t';
            }
            retval+=branchValue;
            retval+="-";
            if (isLeafNode())
                retval+=label;
            else
                retval+=decision.name();
            retval+='\n';
            for (TreeNode branch : branches){
                retval+=branch.stringify(tabLevel+1);
            }
            return retval;
        }
    }
    
    private TreeNode rootNode;
    
    public TreeNode id3Node(Instances i){
        TreeNode treeNode = new TreeNode();
        
        int [] count = calculateCount(i);
        for (int j=0;j<count.length;j++){
            int c = count[j];
            if (c == i.numInstances()){
               treeNode.label = j;
               return treeNode;
            }
        }
        
        if (i.numAttributes()<=1){
            int maxc = -1;
            int maxcj = -1;
            for (int j=0;j<count.length;j++){
                if (count[j]>maxc){
                    maxc=count[j];
                    maxcj = j;
                }
            }
            treeNode.label = maxcj;
            return treeNode;
        }
        
        Attribute bestA = null;
        double bestAIG = -1;
        double entropyOfSet = entropy(i);
        for (int j=0;j<i.numAttributes();j++){
            Attribute a = i.attribute(j);
            if (a!=i.classAttribute()){
                double aIG = infoGain(i,a,entropyOfSet);
                if (aIG>bestAIG){
                    bestAIG = aIG;
                    bestA = a;
                }
            }
        }
        treeNode.decision = bestA;
        Instances [] subSets = splitData(i, bestA);
        for (Instances subSet : subSets){
            if (subSet.numInstances()>0){
                double attributeValue = subSet.firstInstance().value(bestA);
                subSet.deleteAttributeAt(bestA.index());
                TreeNode newBranch = id3Node(subSet);
                newBranch.branchValue = attributeValue;
                treeNode.addBranch(newBranch);
            }else{ 
            }
        }
        return treeNode;
    }
    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
      }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        rootNode = id3Node(i);
    }
    
    @Override
    public double classifyInstance(Instance instance){
        TreeNode nodeIter = rootNode;
        while (!nodeIter.isLeafNode()){
            double decisionAttrVal = instance.value(nodeIter.decision);
            boolean branchFound = false;
            for (TreeNode branch : nodeIter.branches){
                if (branch.branchValue==decisionAttrVal){
                    nodeIter = branch;
                    branchFound = true;
                    break;
                }
            }
            if (!branchFound){
                return 0;
            }
        }
        return nodeIter.label;
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
            if (p!=0)
                result -= Math.log(p)/Math.log(2)*p;
        }
        return result;
    }
    
    public int [] calculateCount(Instances instances){
        
        int numClasses = instances.classAttribute().numValues();
        int numInstances = instances.numInstances();
        int [] num = new int[numClasses];
        for (int i=0;i<numClasses;i++){
            num[i]=0;
        }
        for (int i=0;i<numInstances;i++){
            Instance instance = instances.instance(i);
            double classValue = instance.value(instance.classAttribute());
            int classIndex = (int) classValue;
            num[classIndex]++;
        }
        return num;
    }
    
    public double [] calculateProportion(Instances instances){
        int numClasses = instances.classAttribute().numValues();
        int numInstances = instances.numInstances();
        int [] num = calculateCount(instances);
        double [] result = new double[numClasses];
        for (int i=0;i<numClasses;i++){
            result[i] = ((double)num[i])/numInstances;
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
    
    public String toString(){
        return rootNode.stringify(0);
    }

}
