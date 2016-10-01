/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1;
import Helper.*;
import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author Hp
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here
        String filename = "weather";
        
        //Masih belum mengerti tipe .csv yang dapat dibaca seperti apa
        //CsvToArff convert = new CsvToArff(filename+".csv");
        
        //LOAD FILE
        ArffLoader loader = new ArffLoader();
        File source = new File("src/"+filename+".arff");
//        System.out.println(source.getAbsolutePath());
        loader.setFile(source);
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);
        //END OF LOAD FILE
        
        
        CustomFilter fil = new CustomFilter();
        
        //REMOVE ATTRIBUTE
        //NOT SURE ON HOW TO USE THIS YET
//        structure = fil.removeAttribute(structure);
        
        
        //RESAMPLING
        structure = fil.resampling(structure);
        System.out.println(structure);
        
        
    }
    
}
