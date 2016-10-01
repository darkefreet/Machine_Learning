/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Helper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author ivan
 */
public class AttributeRemover {
    
    /**
    * takes a path to ARFF file as first argument, the index of attribute to remove (in String)
    * as second and thirdly whether to invert or not (true/false).
    */
    public static Instances remove(String filePath, String index) throws IOException, Exception {
        ArffLoader loader2= new ArffLoader();
        loader2.setSource(new File(filePath));
        Instances data2= loader2.getDataSet();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = index;
        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(data2);
        Instances newData2 = Filter.useFilter(data2, remove);
        System.out.println(newData2);
        return newData2;
    }

}
