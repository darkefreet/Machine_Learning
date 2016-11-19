/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Helper;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author Hp
 */
public class CustomFilter {
    public CustomFilter(){
    
    }
    
    public Instances removeAttribute(Instances structure) throws Exception{
        //NORMALIZE AND REMOVE USELESS ATTRIBUTES
        Normalize norm = new Normalize();
        norm.setInputFormat(structure);
        structure = Filter.useFilter(structure, norm);

        RemoveUseless ru = new RemoveUseless();
        ru.setInputFormat(structure);
        structure = Filter.useFilter(structure, ru);
        
        Ranker rank = new Ranker();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        eval.buildEvaluator(structure);
        //END OF NORMALIZATION
        
        return structure;
    }
    
    public Instances resampling(Instances structure){
        Resample filter = new Resample();
	Instances filteredIns = null;
	filter.setBiasToUniformClass(1.0);
	try {
		filter.setInputFormat(structure);
		filter.setNoReplacement(false);
		filter.setSampleSizePercent(100);
		filteredIns = Filter.useFilter(structure, filter);
	} catch (Exception e) {
		e.printStackTrace();
	}
	return filteredIns;
    }
    
    public Instances convertNumericToNominal(Instances structure) throws Exception {
        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]= "1-" + structure.numAttributes();
        convert.setOptions(options);
        convert.setInputFormat(structure);
        structure = Filter.useFilter(structure, convert);
        return structure;
    }
    
    public Instances convertNumericRange(Instances structure) throws Exception{
        for(int i = 0; i<structure.numAttributes()-1;i++){
            if(structure.attribute(i).typeToString(structure.attribute(i)).equals("numeric")){
                structure.sort(i);
                structure = toRange(structure,i);
            }    
        }
        return structure;
    }
    
    //SET ALL VALUES TO THE BATAS BAWAH
    private Instances toRange(Instances structure,int index)throws Exception{
        Attribute attr = structure.attribute(index);
        Attribute classlabel = structure.attribute(structure.numAttributes()-1);
        String label = structure.instance(0).stringValue(classlabel);
        double threshold = structure.instance(0).value(index);
        for(int i = 0; i<structure.numInstances();i++){
            if(!structure.instance(i).stringValue(classlabel).equals(label)){
                label = structure.instance(i).stringValue(classlabel);
                threshold = structure.instance(i).value(index);
            }
            structure.instance(i).setValue(attr, threshold);
        }
        return structure;
    }
    
    public String convertToFit(String value, Instances data, int index){
        int i;
        String threshold = data.attribute(index).value(0);
        for(i = 0; i < data.numDistinctValues(data.attribute(index)) ;i++){
            if(Float.valueOf(value)<Float.valueOf(data.attribute(index).value(i))){
                value = threshold;
                return value;
            }
            threshold = data.attribute(index).value(i);
        }
        value = threshold;
        return value;
    }

}
