package app;

import java.util.Random;

public class Perceptron {
    private Double u;
    private int qtdIn, qtdOut, qtdH;
    private Double[][] wh;
    private Double[][] wo;
    private Double interA = -0.3;
    private Double interB = 0.3;

    public Perceptron(int qtdIn, int qtdOut, int qtdH, Double u) {
        this.u = u;
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;
        this.qtdH = qtdH;
        wh = new Double[qtdIn + 1][qtdH];
        wo = new Double[qtdH + 1][qtdOut];
        
        wh = gerarRandomW(wh);
        wo = gerarRandomW(wo);
    }

    public Double[] learn(Double[] xIn, Double[] y) {
        Double[] x = fill(xIn);
        Double[] hiddenOut = new Double[qtdH + 1];
        for (int j = 0; j < qtdH; j++) {
            hiddenOut[j] = 0d;
            for (int i = 0; i < x.length; i++) {
                hiddenOut[j] += x[i] * wh[i][j];
            }
            hiddenOut[j] = sigmoid(hiddenOut[j]);;
        }
        hiddenOut[qtdH] = 1d;

        Double[] teta = new Double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            teta[j] = 0d;
            for (int i = 0; i < hiddenOut.length; i++) {
                teta[j] = hiddenOut[i] * wo[i][j];
            }
            teta[j] = sigmoid(teta[j]);
        }

        Double[] deltaO = new Double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            deltaO[j] = teta[j] * (1 - teta[j]) * (y[j] - teta[j]);
        }

        Double[] deltaH = new Double[qtdH];
        for (int h = 0; h < qtdH; h++) {
            Double soma = 0d;
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

    public Double[] train(Double[] xIn, Double[] y) {
        Double[] x = fill(xIn);
        Double[] hiddenOut = new Double[qtdH + 1];

        for (int j = 0; j < qtdH; j++) {
            hiddenOut[j] = 0d;
            for (int i = 0; i < x.length; i++) {
                hiddenOut[j] += x[i] * wh[i][j];
            }
            hiddenOut[j] = sigmoid(hiddenOut[j]);;
        }
        hiddenOut[qtdH] = 1d;

        Double[] teta = new Double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            teta[j] = 0d;
            for (int i = 0; i < hiddenOut.length; i++) {
                teta[j] += hiddenOut[i] * wo[i][j];
            }
            teta[j] = sigmoid(teta[j]);
        }
        return teta;
    }

    private Double[] fill(Double[] x) {
        Double[] x_new = new Double[x.length + 1];
        for (int i = 0; i < x.length; i++) {
            x_new[i] = x[i];
        }
        
        x_new[x_new.length - 1] = 1d;
        return x_new;
    }

    private Double[][] gerarRandomW(Double[][] w) {
        Random random = new Random();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = this.interA + (this.interA - this.interB)*random.nextDouble();
            }
        }
        return w;
    }

    private Double sigmoid(Double u) {
        Double value = 1 / (1 + Math.exp(-u));
        return value;
    }
}
