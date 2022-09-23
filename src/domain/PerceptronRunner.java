package domain;

import app.Perceptron;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class PerceptronRunner {
    public static void main(String[] args) throws FileNotFoundException {

        final int QTD_IN = 9;
        final int QTD_OUT = 1;
        final int QTD_H = 9;
        final double U = 0.01;
        final int EPOCA = 100000;

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
        List<double[][][]> listBase = teste();
        double[][][] trainingBase = listBase.get(0);
        double[][][] testBase = listBase.get(1);
        for (int e = 0; e < EPOCA; e++) {
            erroEpTreino = 0;
            erroEpTeste = 0;

            for (int a = 0; a < trainingBase.length; a++) {
                double[] x = trainingBase[a][0];
                double[] y = trainingBase[a][1];
                double[] out = p.learn(x, y);
                erroAmTreino = somador(y, out);
                erroEpTreino += erroAmTreino;
            }
            imprimirTreino(erroEpTreino, e);

            for (int a = 0; a < testBase.length; a++) {
                double[] x = testBase[a][0];
                double[] y = testBase[a][1];
                double[] out = p.train(x, y);
                erroAmTeste = somador(y, out);
                erroEpTeste += erroAmTeste;
            }
            imprimirTeste(erroEpTeste, e);
        }
    }

    public static double somador(double[] y, double[] out) {
        double soma = 0;
        for (int i = 0; i < y.length; i++) {
            soma += Math.abs(y[i] - out[i]);
        }
        return soma;
    }

    public static void imprimirTreino(double erroEp, int epoca) {
        System.out.println("Epoca treino " + (epoca + 1) + "   erro: " + erroEp);
    }
    public static void imprimirTeste(double erroEp, int epoca) {
        System.out.println("Epoca teste " + (epoca + 1) + "   erro: " + erroEp);
    }

    public static double[][][] dataReader() throws FileNotFoundException {
        double[][][] base = new double[286][2][];
        double[] data = new double[9];
        File file = new File("breast-cancer.data");
        Scanner scn = new Scanner(file);
        double out = 0;

        int cont = 0;
        for (int i = 0; i < 286; i++) {
            String linha = scn.nextLine();

            data = inFill(linha);

            base[cont][0] = new double[9];
            for (int j = 0; j < 9; j++) {
                base[cont][0][j] = data[j];
            }
            out = data[9];
            if (out == 0) {
                base[cont][1] = new double[] { 0 };
            } else if (out == 1) {
                base[cont][1] = new double[] { 1 };
            }
            cont++;
        }
        scn.close();
        return base;
    }

    public static List<double[][][]> teste() throws FileNotFoundException {
        double[][][] base = dataReader();
        double[] data = new double[9];
        double[][][] base0 = new double[286][2][9];
        double[][][] base1 = new double[286][2][9];
        double[][][] trainingBase;
        double[][][] testBase;
        double out = 0;

        int cont0 = 0;
        int cont1 = 0;
        for (int i = 0; i < 286; i++) {
            if (base[i][1][0] == 0) {
                for (int j = 0; j < 9; j++) {
                    base0[cont0][0][j] = base[i][0][j];
                }
                base0[cont0][1][0] = base[i][1][0];
                cont0++;
            } else if (base[i][1][0] == 1) {
                for (int j = 0; j < 9; j++) {
                    base1[cont1][0][0] = base[i][0][0];
                }
                base1[cont1][1][0] = base[i][1][0];
                cont1++;
            }
        }
        testBase = new double[Math.abs(cont0 / 4) + Math.abs(cont1 / 4)][2][9];
        int contfor1 = 0;
        for (int i = 0; i < Math.abs(cont0 / 4); i++) {
            for (int j = 0; j < 9; j++) {
                testBase[contfor1][0][j] = base0[i][0][j];
            }
            testBase[contfor1][1] = new double[] { base0[i][1][0] };
            contfor1++;
        }

        for (int i = 0; i < Math.abs(cont1 / 4); i++) {
            for (int j = 0; j < 9; j++) {
                testBase[contfor1][0][j] = base1[i][0][j];
            }
            testBase[contfor1][1] = new double[] { base1[i][1][0] };
            contfor1++;
        }

        List<double[][]> t1 = new ArrayList<double[][]>();
        for (int i = 0; i < testBase.length; i++) {
            t1.add(testBase[i]);  
        }
        Collections.shuffle(t1);
        for (int i = 0; i < contfor1; i++) {
            double[][] d = t1.get(i);
            for (int j = 0; j < 9; j++) {
                testBase[i][0][j] = d[0][j];
            }
            testBase[i][1] = new double[] { d[1][0] };
        }

        int la = (int) (Math.round(cont0 * 0.75) + Math.round(cont1 * 0.75));
        trainingBase = new double[la][2][9];
        int cont2 = 0;
        for (int i = Math.abs(cont0 / 4); i < cont0; i++) {
            for (int j = 0; j < 9; j++) {
                trainingBase[cont2][0][j] = base0[i][0][j];
            }
            trainingBase[cont2][1] = new double[] { base0[i][1][0] };
            cont2++;
        }

        for (int i = Math.abs(cont1 / 4); i < cont1; i++) {
            for (int j = 0; j < 9; j++) {
                trainingBase[cont2][0][j] = base1[i][0][j];
            }
            trainingBase[cont2][1] = new double[] { base1[i][1][0] };
            cont2++;
        }
        List<double[][]> t = new ArrayList<double[][]>();
        for (int i = 0; i < trainingBase.length; i++) {
            t.add(trainingBase[i]);  
        }
        Collections.shuffle(t);
        for (int i = 0; i < cont2; i++) {
            double[][] d = t.get(i);
            for (int j = 0; j < 9; j++) {
                trainingBase[i][0][j] = d[0][j];
            }
            trainingBase[i][1] = new double[] { d[1][0] };
        }
        List<double[][][]> l = new ArrayList<double[][][]>();
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
}
