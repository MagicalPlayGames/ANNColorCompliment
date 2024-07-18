using System;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "Parameters", menuName = "Scriptable Objects/Parameters")]
public class Parameters : ScriptableObject
{
    public byte layers;
    public int[] nodes;
    public float learningRate;
    public byte algoNum;
    public byte outputs;
    public float[][][] weights;
    public float trainingSuccessRate;
    public void setUp(byte l, int[] n, float lR, byte aN, byte o, float[][][] w = null,float tSR = 0)
    {
        layers = l;
        nodes = n;
        learningRate = lR;
        algoNum = aN;
        outputs = o;
        if (w != null)
            weights = w;
        else
            weights = null;
        trainingSuccessRate = tSR;
    }
    public void setUp(Parameters param)
    {
        layers = param.layers;
        nodes = param.nodes;
        learningRate = param.learningRate;
        algoNum = param.algoNum;
        outputs = param.outputs;
        if(param.weights!=null)
            weights = param.weights;
        else
            weights = null;
        trainingSuccessRate = param.trainingSuccessRate;
        
    }
    public void setSuccessRate(float rate)
    {
        trainingSuccessRate = rate;
    }

    public void setWeights(float[][][] w)
    {
        weights = new float[w.Length][][];
        for (int i = 0; i < w.Length; i++)
        {
            weights[i] = new float[w[i].Length][];
            for(int j = 0;j<w[i].Length;j++)
            {
                weights[i][j] = new float[w[i][j].Length];
                for (int z = 0; z < w[i][j].Length; z++)
                    weights[i][j][z] = w[i][j][z];
            }
        }
    }

    public List<float> get1DWeights()
    {
        int size = 0;
        for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
                    size+=weights[i][j].Length;
        List<float> returnWeights = new List<float>();

        for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
                for(int w = 0;w<weights[i][j].Length;w++)
                {
                    returnWeights.Add(weights[i][j][w]);
                }

        return returnWeights;
    }

    public void setWeights(float[] returningWeights)
    {
        int counter = 0;
        weights = new float[layers][][];
        for (int i = 0; i < layers; i++)
        {
            weights[i] = new float[nodes[i]][];
            for (int j = 0; j < nodes[i]; j++)
            {
                if (i == layers - 1)
                {
                    weights[i][j] = new float[1];
                    weights[i][j][0] = returningWeights[counter++];
                }
                else
                {
                    weights[i][j] = new float[nodes[i + 1]];
                    for (int z = 0; z < nodes[i + 1]; z++)
                        weights[i][j][z] = returningWeights[counter++];
                }
            }
        }
    }
}
