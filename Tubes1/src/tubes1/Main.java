/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes1;
import Helper.*;
import java.io.IOException;

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
        csvToArff convert = new csvToArff("weather_csv.csv");
        AttributeRemover.remove("weather.arff", "2");
    }
    
}
