/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1;
import Helper.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import java.util.logging.*;
import tubes1.myClassifiers.myC45;
import tubes1.myClassifiers.myID3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Hp
 */
public class Main {

    public static boolean isNumeric(String s) {  
        return s.matches("[-+]?\\d*\\.?\\d+");  
    }  
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
        String filename = "weather";
        
        //Masih belum mengerti tipe .csv yang dapat dibaca seperti apa
        //CsvToArff convert = new CsvToArff(filename+".csv");
        
        //LOAD FILE
        BufferedReader datafile = readDataFile("src/"+filename+".arff");
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes()-1);
        //END OF LOAD FILE
        
        CustomFilter fil = new CustomFilter();
        
        //REMOVE USELESS ATTRIBUTE
        data = fil.removeAttribute(data);
        System.out.println(data);
        
        Instances[] allData = new Instances[4];
        //data for Id3
        allData[0] = fil.resampling(fil.convertNumericToNominal(data));
        //data for J48
        allData[1] = fil.convertNumericToNominal(fil.resampling(data));
        //data for myId3
        allData[2] = allData[0];        
        //data for myC4.5
        allData[3] = fil.resampling(fil.convertNumericToNominal(fil.convertNumericRange(data)));
        
        data = fil.convertNumericToNominal(data);
        // BUILD CLASSIFIERS
        Classifier[] models = { 
            new Id3(), //C4.5
            new J48(),
            new myID3(),
            new myC45()
        };
        
        for (int j = 0; j < models.length; j++) {
            FastVector predictions = new FastVector();
            //FOR TEN-FOLD CROSS VALIDATION
            Instances[][] split = crossValidationSplit(allData[j], 10);
            // Separate split into training and testing arrays
            Instances[] trainingSplits = split[0];
            Instances[] testingSplits = split[1];
            System.out.println("\n---------------------------------");
            for (int i = 0; i < trainingSplits.length; i++) {
                try {
//                    System.out.println("Building for training Split : " + i);
                    Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);

                    predictions.appendElements(validation.predictions());

                    // Uncomment to see the summary for each training-testing pair.
//                    System.out.println(models[j].toString());
                } catch (Exception ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
                // Calculate overall accuracy of current classifier on all splits
                double accuracy = calculateAccuracy(predictions);

                // Print current classifier's name and accuracy in a complicated,
                // but nice-looking way.
                System.out.println(String.format("%.2f%%", accuracy)
                );
            }
            models[j].buildClassifier(allData[j]);
            Model.save(models[j],models[j].getClass().getSimpleName());
        }
        
        
        //test instance
        Instances trainingSet = new Instances("Rel", getFvWekaAttributes(data), 10);
        trainingSet.setClassIndex(data.numAttributes() - 1);

        Instance testInstance = new Instance(data.numAttributes());
        for(int i = 0; i<data.numAttributes()-1;i++){
            System.out.print("Masukkan "+ data.attribute(i).name()+" : ");
            Scanner in = new Scanner(System.in);
            String att = in.nextLine();
            if(isNumeric(att)){
                att = fil.convertToFit(att, data, i);
            }
            testInstance.setValue(data.attribute(i),att);
        }

//        System.out.println(testInstance);
//        System.out.println(testInstance.classAttribute().index());
    
    
        trainingSet.add(testInstance);

        Classifier Id3 = Model.load("Id3");
        Classifier J48 = Model.load("J48");
        Classifier myID3 = Model.load("myID3");
        Classifier MyC45 = Model.load("myC45");
//        Classifier MyId3 = Model.load("myID3");

        Instance A = trainingSet.instance(0);
        Instance B = trainingSet.instance(0);
        Instance C = trainingSet.instance(0);
        Instance D = trainingSet.instance(0);
        
//test with ID3 WEKA
        A.setClassValue(Id3.classifyInstance(trainingSet.instance(0)));
        System.out.println("Id3 Weka : " + A);
        
        //test with C4.5 WEKA
        B.setClassValue(J48.classifyInstance(trainingSet.instance(0)));
        System.out.println("C4.5 Weka : " + B);
        
        //test with my C4.5
        C.setClassValue(MyC45.classifyInstance(trainingSet.instance(0)));
        System.out.println("My C4.5 : " + C);
        
        //test with my ID3
        D.setClassValue(myID3.classifyInstance(trainingSet.instance(0)));
        System.out.println("My ID3 : " + D);
    }
    
    private static FastVector getFvWekaAttributes(Instances data) {
        int numAttributes = data.numAttributes();
        FastVector fvWekaAttributes = new FastVector(numAttributes);
        for (int i = 0; i < numAttributes; i++) {
            fvWekaAttributes.addElement(data.attribute(i));
        }
        return fvWekaAttributes;
    }
    
}
