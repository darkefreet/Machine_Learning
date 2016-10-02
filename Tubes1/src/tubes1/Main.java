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
import java.util.logging.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.SimpleCart;
import weka.core.FastVector;
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
    
    public static Evaluation classify(Classifier model,Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);
        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);
        return evaluation;
    }
    
    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;
        for (int i = 0; i < predictions.size(); i++) {
                NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
                if (np.predicted() == np.actual()) {
                        correct++;
                }
        }
        return 100 * correct / predictions.size();
    }
    
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here
        CsvToArff convert = new CsvToArff("weather_csv.csv");
        String filename = "weather";
        
        //Masih belum mengerti tipe .csv yang dapat dibaca seperti apa
        //CsvToArff convert = new CsvToArff(filename+".csv");
        
        //LOAD FILE
        BufferedReader datafile = readDataFile("src/"+filename+".arff");
        Instances data = new Instances(datafile);
        data.setClassIndex(0);
        //END OF LOAD FILE
        
        CustomFilter fil = new CustomFilter();
        data = fil.convertNumericToNominal(data);
        System.out.println(data);
        
        //REMOVE USELESS ATTRIBUTE
        data = fil.removeAttribute(data);
        
        //RESAMPLING
        data = fil.resampling(data);
        //System.out.println(data);
        
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
        
        for (int j = 0; j < models.length; j++) {
            FastVector predictions = new FastVector();
            for (int i = 0; i < trainingSplits.length; i++) {
                try {
                    System.out.println("Building for training Split : " + i);
                    Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);

                    predictions.appendElements(validation.predictions());

                    // Uncomment to see the summary for each training-testing pair.
                    System.out.println(models[j].toString());
                } catch (Exception ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
                // Calculate overall accuracy of current classifier on all splits
                double accuracy = calculateAccuracy(predictions);

                // Print current classifier's name and accuracy in a complicated,
                // but nice-looking way.
                System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
                                + String.format("%.2f%%", accuracy)
                                + "\n---------------------------------");
            }
        }
    }
}
