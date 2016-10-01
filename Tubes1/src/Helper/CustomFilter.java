/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Helper;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.supervised.instance.Resample;

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
}
