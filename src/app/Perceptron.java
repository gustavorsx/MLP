package app;

import java.util.Random;

public class Perceptron {
    private double u;
    private int qtdIn, qtdOut, qtdH;
    private double[][] wh;
    private double[][] wo;
    private double interA = -0.3;
    private double interB = 0.3;

    public Perceptron(int qtdIn, int qtdOut, int qtdH, double u) {
        this.u = u;
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;
        this.qtdH = qtdH;
        wh = new double[qtdIn + 1][qtdH];
        wo = new double[qtdH + 1][qtdOut];
        
        wh = gerarRandomW(wh);
        wo = gerarRandomW(wo);
    }

    public double[] learn(double[] xIn, double[] y) {
        double[] x = fill(xIn);
        double[] hiddenOut = new double[qtdH + 1];

        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                hiddenOut[j] += x[i] * wh[i][j];
            }
            hiddenOut[j] = sigmoid(hiddenOut[j]);;
        }
        hiddenOut[qtdH] = 1;

        double[] teta = new double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < hiddenOut.length; i++) {
                teta[j] += hiddenOut[i] * wo[i][j];
            }
            teta[j] = sigmoid(teta[j]);
        }

        double[] deltaO = new double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            deltaO[j] = teta[j] * (1 - teta[j]) * (y[j] - teta[j]);
        }

        double[] deltaH = new double[qtdH];
        for (int h = 0; h < qtdH; h++) {
            double soma = 0;
            for(int j = 0; j < qtdOut; j++) {
                soma += deltaO[j] * wo[h][j];
            };

            deltaH[h] = hiddenOut[h] * (1 - hiddenOut[h]) * soma;
        }

        // Ajuste dos pesos da camada intermediária
        // peso WHij += ni * deltaH[j] * xi; (Dois for aninhados -> pra i e j)
        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                wh[i][j] += u * deltaH[j] * x[i];
            }
        }

        // Ajuste dos pesos da saída
        // peso WTETAhj += ni * deltaTetaj * Hh; (Dois for aninhados -> pra h e j)
        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < hiddenOut.length; i++) {
                wo[i][j] += u * deltaO[j] * hiddenOut[i];
            }
        }

        return teta;
    }

    public double[] train(double[] xIn, double[] y) {
        double[] x = fill(xIn);
        double[] hiddenOut = new double[qtdH + 1];

        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                hiddenOut[j] += x[i] * wh[i][j];
            }
            hiddenOut[j] = sigmoid(hiddenOut[j]);;
        }
        hiddenOut[qtdH] = 1;

        double[] teta = new double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < hiddenOut.length; i++) {
                teta[j] += hiddenOut[i] * wo[i][j];
            }
            teta[j] = sigmoid(teta[j]);
        }
        return teta;
    }

    private double[] fill(double[] x) {
        double[] x_new = new double[x.length + 1];
        for (int i = 0; i < x.length; i++) {
            x_new[i] = x[i];
        }
        
        x_new[x_new.length - 1] = 1;
        return x_new;
    }

    private double[][] gerarRandomW(double[][] w) {
        Random random = new Random();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = random.nextDouble(this.interA, this.interB);
            }
        }
        return w;
    }

    private double sigmoid(double u) {
        double value = 1 / (1 + Math.exp(-u));
        return value;
    }
}
