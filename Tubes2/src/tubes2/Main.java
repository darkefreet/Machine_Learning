/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import tubes2.myClusterers.myAgnes;
import tubes2.myClusterers.myKMeans;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 *
 * @author nim_13512501
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
    
    public static void normalizeData(Instances data, ArrayList<Integer> numericIndex){
        ArrayList<Double> mean = new ArrayList();
        ArrayList<Double> standardDef = new ArrayList();
        
        //INISIALISASI
        for(int i = 0;i<numericIndex.size();i++){
            mean.add(0.0);
            standardDef.add(0.0);
        }
        //HITUNG MEAN
        for(int i = 0; i<data.numInstances();i++){
            for(int j = 0;j<numericIndex.size();j++){
                mean.set(j,(mean.get(j)+data.instance(i).value(numericIndex.get(j))/data.numInstances()));
            }
        }
        
        //HITUNG STANDAR DEVIASI
        for(int i = 0; i<data.numInstances();i++){
            for(int j = 0;j<numericIndex.size();j++){
                standardDef.set(j,standardDef.get(j)+((Math.abs(data.instance(i).value(numericIndex.get(j))-mean.get(j)))/data.numInstances()));
            }
        }
        
        //UBAH NILAI INSTANCE MENJADI Z-SCORE
        for(int i = 0; i<data.numInstances();i++){
            for(int j = 0;j<numericIndex.size();j++){
                data.instance(i).setValue(numericIndex.get(j),(data.instance(i).value(numericIndex.get(j)) - mean.get(j))/standardDef.get(j));
            }
        }
    }
    
    public static boolean isNumeric(String str)  
    {  
      try  
      {  
        double d = Double.parseDouble(str);  
      }  
      catch(NumberFormatException nfe)  
      {  
        return false;  
      }  
      return true;  
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
        BufferedReader datafile = readDataFile("data/"+filename+".arff");
        Instances data = new Instances(datafile);
        
        ArrayList<Integer> numericIdx = new ArrayList<Integer>();
        for (int i = 0;i < data.numAttributes();i++){
            if(data.attribute(i).isNumeric()){
                numericIdx.add(i);
            }
        }
        System.out.println();
        System.out.println("\n----SEBELUM NORMALISASI-----");
        System.out.println(data);
        normalizeData(data,numericIdx);
        System.out.println("\n----SETELAH NORMALISASI-----");
        System.out.println(data);
        
        //END OF LOAD FILE
        
        SimpleKMeans simpleK = new SimpleKMeans();
        simpleK.setNumClusters(4);
        Clusterer [] clusterers = {
            simpleK,
            new myKMeans(4),
            new myAgnes(4)
        };
        
        boolean first = true;
        for (Clusterer clusterer: clusterers){
            try {
                clusterer.buildClusterer(data);
                System.out.println("\n\n----HASIL CLUSTERING-----");
                System.out.println(clusterer);
            } catch (Exception ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
}
