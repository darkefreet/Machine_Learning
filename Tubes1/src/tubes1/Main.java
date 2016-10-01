/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1;
import Helper.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.SimpleCart;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author Hp
 */
public class Main {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;
        try {
                inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
                System.err.println("File not found: " + filename);
        }
        return inputReader;
    }
    
    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
            Instances[][] split = new Instances[2][numberOfFolds];
            for (int i = 0; i < numberOfFolds; i++) {
                    split[0][i] = data.trainCV(numberOfFolds, i);
                    split[1][i] = data.testCV(numberOfFolds, i);
            }
            return split;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here
        csvToArff convert = new csvToArff("weather_csv.csv");
        String filename = "weather";
        
        //Masih belum mengerti tipe .csv yang dapat dibaca seperti apa
        //CsvToArff convert = new CsvToArff(filename+".csv");
        
        //LOAD FILE
        BufferedReader datafile = readDataFile("src/"+filename+".arff");
        Instances data = new Instances(datafile);
        data.setClassIndex(0);
        //END OF LOAD FILE
        
        CustomFilter fil = new CustomFilter();
        
        //REMOVE USELESS ATTRIBUTE
        data = fil.removeAttribute(data);
        
        //RESAMPLING
        data = fil.resampling(data);
        System.out.println(data);
        
        //FOR TEN-FOLD CROSS VALIDATION
        Instances[][] split = crossValidationSplit(data, 10);
        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // BUILD CLASSIFIERS
        Classifier[] models = { 
            new J48(), //C4.5
            new Id3() //ID3
        };
    }
    
}
