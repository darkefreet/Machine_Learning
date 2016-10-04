/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1.myClassifiers;

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
public class myID3 extends Classifier {
    
    private class TreeNode{
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
    }
    
    private TreeNode rootNode;
    
    public TreeNode id3Node(Instances i){
        TreeNode treeNode = new TreeNode();
        
        int [] count = calculateCount(i);
        for (int c : count){
            if (count[c] == i.numInstances()){
               treeNode.label = c;
               return treeNode;
            }
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
                treeNode.addBranch(newBranch);
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
                return -1;
            }
        }
        return nodeIter.label;
    }
    
    public double infoGain(Instances instances, Attribute attribute, double entropyOfSet){
        return 0;
    }
    
    public double entropy(Instances instances){
        double result = 0;
        double[] proportion = calculateProportion(instances);
        for (double p : proportion){
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
            int classIndex = instance.classIndex();
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
            result[i] = num[i]/numInstances;
        }
        return result;
    }
}
