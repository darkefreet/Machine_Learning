/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Helper;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Hp
 */
public class csvToArff {
    public csvToArff(String filename) throws IOException{
        // load CSV
        CSVLoader loader = new CSVLoader();
        String source = "../"+filename;
        URL url = getClass().getResource(source);
        loader.setSource(new File(url.getPath()));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        String newfile = "src/"+filename.substring(0, filename.lastIndexOf('.'))+".arff";
        saver.setFile(new File(newfile));
        saver.writeBatch();
    }
}
