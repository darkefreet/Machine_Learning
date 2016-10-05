/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1.myClassifiers;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
 */
public class myC45 extends Classifier {

    protected PruneableClassifierTree m_root;
    private boolean isLeaf = false;
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        ModelSelection mod;
        int numSets = 10;
        int randomSeed = 1;
        
        mod = new C45ModelSelection(2,i);
        
        //test the capabilities
        getCapabilities().testWithFail(i);
        i = new Instances(i);
        i.deleteWithMissingClass();
        
        Random random = new Random(randomSeed);
        i.stratify(numSets);
        
        ClassifierSplitModel localMod = mod.selectModel(i.trainCV(numSets, numSets - 1, random),i.testCV(numSets, numSets - 1));
        Distribution m_test = new Distribution(i.testCV(numSets, numSets - 1), localMod);
        Instances[] localTrain,localTest;
        Instances train,test;
        PruneableClassifierTree[] m_sons;
        m_sons = new PruneableClassifierTree [localMod.numSubsets()];
        if (localMod.numSubsets() > 1) {
            localTrain = localMod.split(i.trainCV(numSets, numSets - 1, random));
            localTest = localMod.split(i.testCV(numSets, numSets - 1));
            train = i.trainCV(numSets, numSets - 1, random);
            test = i.testCV(numSets, numSets - 1);
            for (int j=0;j<m_sons.length;j++) {
                PruneableClassifierTree newTree = new PruneableClassifierTree(mod,true,10,true,1);
                newTree.buildTree(train, test, false);
                m_sons[j] = newTree;
                localTrain[j] = null;
                localTest[j] = null;
            }
        }else{
            isLeaf = true;
        }
        
        m_root = m_sons[m_sons.length-1];
        m_root.prune();
        
        ((C45ModelSelection)mod).cleanup();
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        return m_root.classifyInstance(instance);
    }
    
}
