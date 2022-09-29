package domain;

import app.Perceptron;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class PerceptronRunner {
    public static void main(String[] args) throws FileNotFoundException, IOException {

        final int QTD_IN = 9;
        final int QTD_OUT = 1;
        final int QTD_H = 9;
        final double U = 0.01;
        final int EPOCA = 10000;

        /* AND */
        // final double[][][] DATABASE = {
        // { { 0D, 0D }, { 0D } },
        // { { 0D, 1D }, { 0D } },
        // { { 1D, 0D }, { 0D } },
        // { { 1D, 1D }, { 1D } }
        // };

        // /* XOR */
        // final double[][][] DATABASE = {
        // { { 0D, 0D }, { 0D } },
        // { { 0D, 1D }, { 1D } },
        // { { 1D, 0D }, { 1D } },
        // { { 1D, 1D }, { 0D } }
        // };

        /* OR */
        // final double[][][] DATABASE = {
        // { { 0D, 0D }, { 0D } },
        // { { 0D, 1D }, { 1D } },
        // { { 1D, 0D }, { 1D } },
        // { { 1D, 1D }, { 1D } }
        // };

        /* Robô */
        // final double[][][] DATABASE = {
        // { { 0D, 0D, 0D }, { 1D, 1D } },
        // { { 0D, 0D, 1D }, { 0D, 1D } },
        // { { 0D, 1D, 0D }, { 1D, 0D } },
        // { { 0D, 1D, 1D }, { 0D, 1D } },
        // { { 1D, 0D, 0D }, { 1D, 0D } },
        // { { 1D, 0D, 1D }, { 1D, 0D } },
        // { { 1D, 1D, 0D }, { 1D, 0D } },
        // { { 1D, 1D, 1D }, { 1D, 0D } }
        // };

        Perceptron p = new Perceptron(QTD_IN, QTD_OUT, QTD_H, U);
        
        double erroEpTreino = 0;
        double erroAmTreino = 0;
        double erroEpTeste = 0;
        double erroAmTeste = 0;
        List<List<Double[][]>> listBase = teste();
        List<Double[][]> trainingBase = listBase.get(0);
        List<Double[][]> testBase = listBase.get(1);
        List<String> str = new ArrayList<String>();
        for (int e = 0; e < EPOCA; e++) {
            erroEpTreino = 0;
            erroEpTeste = 0;

            for (int a = 0; a < trainingBase.size(); a++) {
                Double[] x = trainingBase.get(a)[0];
                Double[] y = trainingBase.get(a)[1];
                Double[] out = p.learn(x, y);
                erroAmTreino = somador(y, out);
                erroEpTreino += erroAmTreino;
            }

            for (int a = 0; a < testBase.size(); a++) {
                Double[] x = testBase.get(a)[0];
                Double[] y = testBase.get(a)[1];
                Double[] out = p.train(x, y);
                erroAmTeste = somador(y, out);
                erroEpTeste += erroAmTeste;
            }
            str.add(e + " " + erroEpTreino + " " + erroEpTeste);
            imprimirTeste(erroEpTreino, erroEpTeste, e);
        }
        dataWriter(str);
    }

    public static Double somador(Double[] y, Double[] out) {
        Double soma = 0d;
        for (int i = 0; i < y.length; i++) {
            soma += Math.abs(y[i] - out[i]);
        }
        return soma;
    }

    public static void imprimirTeste(double erroEpTreino, double erroEpTeste, int epoca) {
        System.out.println("Epoca teste " + (epoca + 1) + "   erro: " + erroEpTreino + " " + erroEpTeste);
    }

    public static List<List<Double[][]>> dataReader() throws FileNotFoundException {
        List<Double[][]> base0 = new ArrayList<Double[][]>();
        List<Double[][]> base1 = new ArrayList<Double[][]>();
        List<List<Double[][]>> bases = new ArrayList<List<Double[][]>>();
        double[] data = new double[9];
        File file = new File("src/breast-cancer.data");
        Scanner scn = new Scanner(file);
        double out = 0;

        int cont0 = 0;
        int cont1 = 0;
        for (int i = 0; i < 286; i++) {
            String linha = scn.nextLine();

            data = inFill(linha);

            out = data[9];
            if (out == 0) {
                var n = new Double[2][];
                n[0] = new Double[9];
                n[1] = new Double[] { 0d };
                for (int j = 0; j < 9; j++) {
                    n[0][j] = data[j];
                }
                base0.add(n);
            } else if (out == 1) {
                var n = new Double[2][];
                n[0] = new Double[9];
                n[1] = new Double[] { 1d };
                for (int j = 0; j < 9; j++) {
                    n[0][j] = data[j];
                }
                base1.add(n);
            }
            cont0++;
            cont1++;
        }
        bases.add(base0);
        bases.add(base1);
        scn.close();
        return bases;
    }

    public static List<List<Double[][]>> teste() throws FileNotFoundException {
        List<List<Double[][]>> basesList = dataReader();
        int trainingBase0Size = basesList.get(0).size() * 3 / 4;
        int testBase0Size = basesList.get(0).size() * 1 / 4;
        int trainingBase1Size = basesList.get(1).size() * 3 / 4;
        int testBase1Size = basesList.get(1).size() * 1 / 4;

        if ((testBase0Size + trainingBase0Size) < basesList.get(0).size())
            trainingBase0Size++;

        if ((testBase1Size + trainingBase1Size) < basesList.get(1).size())
            trainingBase1Size++;

        List<Double[][]> trainingBase = new ArrayList<Double[][]>();
        List<Double[][]> testBase = new ArrayList<Double[][]>();

        Collections.shuffle(basesList.get(0));
        Collections.shuffle(basesList.get(1));

        for (int i = 0; i < trainingBase0Size; i++) {
            var t = basesList.get(0).get(0);
            trainingBase.add(t);
            basesList.get(0).remove(0);
        }
        for (int i = 0; i < trainingBase1Size; i++) {
            var t = basesList.get(1).get(0);
            trainingBase.add(t);
            basesList.get(1).remove(0);
        }

        for (int i = 0; i < testBase0Size; i++) {
            var t = basesList.get(0).get(0);
            testBase.add(t);
            basesList.get(0).remove(0);
        }
        for (int i = 0; i < testBase1Size; i++) {
            var t = basesList.get(1).get(0);
            testBase.add(t);
            basesList.get(1).remove(0);
        }

        List<List<Double[][]>> l = new ArrayList<List<Double[][]>>();
        l.add(trainingBase);
        l.add(testBase);
        return l;
    }

    public static double[] inFill(String linha) {
        double[] entradas;
        entradas = Arrays.stream(linha.split(" ")).map(String::trim)
                .mapToDouble(Double::parseDouble).toArray();
        return entradas;
    }

    public static void dataWriter(List<String> str) throws IOException {
        FileWriter fwEpoca = new FileWriter("erros.txt");
        PrintWriter printWriterEpoca = new PrintWriter(fwEpoca);
        for (String string : str) {
            printWriterEpoca.println(string);;
        }
        printWriterEpoca.close();
    }

}
