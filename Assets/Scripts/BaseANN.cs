using UnityEngine;

using System;

namespace MatchingColorPicker
{
    class BaseANN
    {
        
        //Runs per operate action
        private byte algorithmChoice;

        //Neuron
        [Serializable]
        public struct node
        {
            //Calc for output
            public float curSum;
            //Store for later
            public float previousSum;
            //Output
            public float output;
            //current Weights from previous layer to this node
            public float[] currentWeights;
            //adjustment storage for weight backward propergation
            public float[] errorAdjustments;
        };
        //Layer of Neurons
        [Serializable]
        public struct layer
        {
            public node[] nodes;
        }

        //Brain with layers
        [Serializable]
        public struct brain
        {
            //Input
            public layer input;
            //Hidden Layers
            public layer[] layers;
            //Output (change to multiple outputs)
            public layer outputs;
            //Expected output
            public layer realOutputs;
            public float learningRate;
        }

        public brain control;

        private Parameters param;

        //This ANN calculates the probability a random number is divisble by divisbleBy

        //Constructor 1
        public void setUpNetwork(byte layers, int[] nodes, float learningRate, byte algoNum, byte outputs)
        {
            param = (Parameters)ScriptableObject.CreateInstance<Parameters>();
            param.setUp(layers, nodes, learningRate, algoNum, outputs);
            this.algorithmChoice = algoNum;
            control = new brain();
            control.learningRate = learningRate;
            this.control.outputs.nodes = new node[outputs];
            control.layers = new layer[layers];
            for (int i = 0; i < layers; i++)
            {
                control.layers[i].nodes = new node[nodes[i]];
                for (int j = 0; j < nodes[i]; j++)
                {
                    node newestNode = new node();
                    newestNode.curSum = newestNode.output = 0;
                    if (i == layers - 1)
                        newestNode.currentWeights = new float[1];
                    else
                        newestNode.currentWeights = new float[nodes[i + 1]];
                    newestNode.errorAdjustments = new float[newestNode.currentWeights.Length];
                    control.layers[i].nodes[j] = newestNode;
                }
            }
            resetWeights();
        }

        //Constructor 2
        public void setUpNetwork(Parameters parameters)
        {

            param = (Parameters)ScriptableObject.CreateInstance<Parameters>();
            param.setUp(parameters);
            algorithmChoice = parameters.algoNum;
            control = new brain();
            control.learningRate = parameters.learningRate;
            control.outputs.nodes = new node[parameters.outputs];
            control.layers = new layer[parameters.layers];
            for (int i = 0; i < parameters.layers; i++)
            {
                control.layers[i].nodes = new node[parameters.nodes[i]];
                for (int j = 0; j < parameters.nodes[i]; j++)
                {
                    node newestNode = new node();
                    newestNode.curSum = newestNode.output = 0;
                    if (i == parameters.layers - 1)
                        newestNode.currentWeights = new float[1];
                    else
                        newestNode.currentWeights = new float[parameters.nodes[i + 1]];
                    newestNode.errorAdjustments = new float[newestNode.currentWeights.Length];
                    control.layers[i].nodes[j] = newestNode;
                }
            }
            if(parameters.weights!=null)
                if(parameters.weights.Length >0)
                {
                    setWeights(parameters.weights);
                    return;
                }
            resetWeights();
        }
        //Overall Processing
        public bool processData(bool testingCase, float[][] inputs, float[] expectedOutcome = null)
        {
            readInput(inputs);
            writeOutputs();
            if (testingCase)
            {
                readTrueOutput(expectedOutcome);
                return (Math.Abs(averageError(control.outputs, control.realOutputs)) < 0.3f);
            }
            return true;
        }

        //Set all weights to random variables
        public void resetWeights()
        {
            System.Random rnd = new System.Random();
            for (int i = 0; i < control.layers.Length; i++)
            {
                for (int j = 0; j < control.layers[i].nodes.Length; j++)
                {
                    for (int w = 0; w < control.layers[i].nodes[j].currentWeights.Length; w++)
                    {
                        control.layers[i].nodes[j].currentWeights[w] = (float)rnd.NextDouble();
                    }
                }
            }
        }


        //reads 10101 string input and puts it into the input layer, setting all input weights to 1
        public void readInput(float[][] input)
        {
            control.input.nodes = new node[input.Length];
            for (int i = 0; i < (input.Length); i ++)
            {
                control.input.nodes[i] = new node();
                control.input.nodes[i].output = input[i][0];
                control.input.nodes[i].currentWeights = new float[control.layers[0].nodes.Length];
            }
            for (int i = 0; i < control.input.nodes.Length; i++)
                for (int j = 0; j < control.layers[0].nodes.Length; j++)
                    control.input.nodes[i].currentWeights[j] = 1.0f;
        }

        public void readTrueOutput(float[] output)
        {
            control.realOutputs = new layer();
            control.realOutputs.nodes = new node[1];
            control.realOutputs.nodes[0].output = output[0];
        }

        //Forward Proprogation
        public float[] writeOutputs()
        {
            float finalOutput = 0;
            bool inputFlag = true;
            int layersLength = control.layers.Length;
            for (int i = 0; i < layersLength; i++)
            {
                layer curLayer = control.layers[i];
                layer previousLayer;
                if (inputFlag)
                {
                    previousLayer = control.input;
                    inputFlag = false;
                }
                else
                {
                    previousLayer = control.layers[i - 1];
                }
                for (int j = 0; j < curLayer.nodes.Length; j++)
                {
                    node curNode = curLayer.nodes[j];
                    for (int w = 0; w < previousLayer.nodes.Length; w++)
                    {
                        node previousNode = previousLayer.nodes[w];
                        curNode.curSum += previousNode.output * previousNode.currentWeights[0];
                    }
                    control.layers[i].nodes[j].previousSum = curNode.curSum;
                    control.layers[i].nodes[j].output = sigmoid(curNode.curSum);
                    control.layers[i].nodes[j].curSum = 0;
                }
            }
            float[] finalOutputs = new float[control.outputs.nodes.Length];
            for (int j = 0; j < control.outputs.nodes.Length; j++)
            {
                for (int i = 0; i < control.layers[layersLength - 1].nodes.Length; i++)
                {
                    finalOutput += control.layers[layersLength - 1].nodes[i].output * control.layers[layersLength - 1].nodes[i].currentWeights[j];
                }
                finalOutputs[j] = tanh(finalOutput);
                control.outputs.nodes[j].output = finalOutputs[j];
                finalOutput = 0;
            }
            return finalOutputs;
        }

        private float[] topErrors(layer output, layer realOutput)
        {
            float[] errors = new float[output.nodes.Length];
            for (int i = 0; i < errors.Length; i++)
            {
                errors[i] = realOutput.nodes[i].output - output.nodes[i].output;
            }
            return errors;
        }

        private float averageError(layer output, layer realOutput)
        {
            float sum = 0;
            for (int i = 0; i < output.nodes.Length; i++)
            {
                sum += realOutput.nodes[i].output - output.nodes[i].output;
            }
            return sum / ((float)output.nodes.Length);
        }

        //Updates the weights
        //Algorithm 0: findSingleError(),findMatrixErrors
        //Algorithm 1: findSingleError(), errorTotalRespectCurWeight()
        //Algorithm 2: topError(),hiddenErrors()
        public void updateAllWeights()
        {
            for (int i = control.layers.Length - 1; i > -1; i--)
            {
                layer curLayer = control.layers[i];
                for (int j = 0; j < curLayer.nodes.Length; j++)
                {
                    node curNode = curLayer.nodes[j];
                    float[] errorAdjust;
                    if (i == control.layers.Length - 1)
                    //output layer to last hidden layer
                    {
                        errorAdjust = topErrors(control.outputs, control.realOutputs);
                    }
                    else
                    {
                        //layer to layer
                        if (algorithmChoice == 0)
                            errorAdjust = findMatrixErrors(curLayer.nodes, control.layers[i + 1].nodes);
                        else if (algorithmChoice == 1)
                            errorAdjust = errorTotalRespectCurWeight(curNode, control.layers[i + 1].nodes);
                        else
                            errorAdjust = hiddenErrors(curLayer, control.layers[i + 1]);

                    }

                    control.layers[i].nodes[j].errorAdjustments = errorAdjust;

                    for (int w = 0; w < curNode.currentWeights.Length; w++)
                    {
                        //edit weights
                        control.layers[i].nodes[j].currentWeights[w] -= errorAdjust[w] * control.learningRate;
                    }
                }
            }
        }

        //e^x/(1+e^x)
        private float sigmoid(float x)
        {
            float EX = (float)Math.Exp(x);
            return EX / (1 + EX);
        }

        //(e^-x)/((1+e^-x)^2)
        private float sigmoidPrime(float x)
        {
            float negativeEX = (float)Math.Exp(-x);
            return negativeEX / ((float)Math.Pow(1 + negativeEX, 2));
        }


        //Errors = previousError*sigmoidPrime(non-activated output)DOT(currentWeights^T)DOT(upperInputs^T)
        private float[] hiddenErrors(layer curLayer, layer upperLayer)
        {
            //hidden errors
            float[][] previousErrors = new float[upperLayer.nodes.Length][];
            for (int i = 0; i < previousErrors.Length; i++)
            {
                previousErrors[i] = new float[upperLayer.nodes[i].errorAdjustments.Length];
                for (int j = 0; j < previousErrors[i].Length; j++)
                {
                    previousErrors[i][j] = upperLayer.nodes[i].errorAdjustments[j];
                }
            }

            float[][] sigmoidPrimes = new float[1][];
            sigmoidPrimes[0] = new float[upperLayer.nodes.Length];
            for (int i = 0; i < sigmoidPrimes[0].Length; i++)
            {
                sigmoidPrimes[0][i] = sigmoidPrime(upperLayer.nodes[i].previousSum);
            }
            float[][] weightsTransposed = new float[upperLayer.nodes.Length][];
            for (int i = 0; i < weightsTransposed.Length; i++)
            {
                weightsTransposed[i] = upperLayer.nodes[i].currentWeights;
            }
            weightsTransposed = transpose(weightsTransposed);

            float[][] inputsTransposed = new float[1][];
            inputsTransposed[0] = new float[curLayer.nodes.Length];
            for (int i = 0; i < inputsTransposed[0].Length; i++)
            {
                inputsTransposed[0][i] = curLayer.nodes[i].output;
            }

            inputsTransposed = transpose(inputsTransposed);

            float[][] products = new float[1][];
            products[0] = dotProduct(sigmoidPrimes, weightsTransposed);
            products[0] = dotProduct(products, inputsTransposed);
            return dotProduct(products, previousErrors);
        }


        //The Total Error in respect to the Current Output
        private float errorTotalRespectCurOutput(float outcome, float trueOutcome)
        {
            return -(trueOutcome - outcome);
        }

        //The Current Output in respect to the Net Output
        private float curOutputRespectNetOutput(float outcome)
        {
            return outcome * (1 - outcome);
        }


        //The Net Output in respect to the current weights

        private float netOutputRespectCurWeights(float weight, float outcome)
        {
            return weight * outcome;
        }

        //The Total Error in respect to the current weights
        //The Net Output in respect to the current weights * The Total Error in respect to the Current Output * The Current Output in respect to the Net Output
        private float[] errorTotalRespectCurWeight(node curNode, node[] upperLayer)
        {
            //https://www.edureka.co/blog/backpropagation/
            //Just read it
            float[] errors = new float[curNode.currentWeights.Length];
            for (int i = 0; i < errors.Length; i++)
            {
                float errorTotal = 0;
                for (int j = 0; j < upperLayer[i].errorAdjustments.Length; j++)
                    errorTotal += upperLayer[i].errorAdjustments[j];
                float curOutput = upperLayer[i].previousSum;

                float netOutput = upperLayer[i].output;
                float curWeight = curNode.currentWeights[i];
                errors[i] = errorTotalRespectCurOutput(curOutput, errorTotal) * curOutputRespectNetOutput(netOutput) * netOutputRespectCurWeights(curWeight, netOutput);
            }
            return errors;
        }



        //(e^(x) - e^(-x)) / (e^(x) + e^(-x))
        private float tanh(float x)
        {
            float EX = (float)Math.Exp(x);
            float negativeEX = (float)Math.Exp(-x);
            return (EX - negativeEX) / (EX + negativeEX);
        }

        //Mean Square Error
        private float errorMSE(float outcome, float trueOutcome)
        {
            return (float)Math.Pow(trueOutcome - outcome, 2);
        }

        //Error1 = w1/totalWeights * error
        private float[] findSingleError(float[] weights, float error)
        {
            float totalWeights = 0;
            for (int i = 0; i < weights.Length; i++)
                totalWeights += weights[i];
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (weights[i] / totalWeights) * error;

            return weights;
        }

        //Errors = weights * previousErrors
        private float[] findMatrixErrors(node[] curLayer, node[] upperLayer)
        {
            //curLayer for weights
            //upperLayer for errors

            //weights: [node][weight]
            //errors: [weights of node][node]


            float[][] weights = new float[curLayer.Length][];
            for (int i = 0; i < weights.Length; i++)
                weights[i] = curLayer[i].currentWeights;

            float[][] errors = new float[upperLayer[0].currentWeights.Length][];
            for (int i = 0; i < errors.Length; i++)
            {
                errors[i] = new float[upperLayer.Length];
                for (int j = 0; j < errors[i].Length; j++)
                {
                    errors[i][j] = upperLayer[j].errorAdjustments[i];
                }
            }
            errors = transpose(errors);
            return dotProduct(errors, weights);
        }

        //Dot product of two matricies
        private float[] dotProduct(float[][] errors, float[][] weights)
        {
            int size;
            if (errors.Length > weights.Length)
                size = errors.Length;
            else
                size = weights.Length;

            float[] outcomes = new float[size];

            for (int i = 0; i < size; i++)
            {
                outcomes[i] = 0;
                if (size == errors.Length)
                {
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        if (i >= weights.Length || j >= errors[i].Length)
                            break;
                        for (int w = 0; w < weights[i].Length; w++)
                        {
                            outcomes[i] += weights[i][w] * errors[i][j];
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < weights[i].Length; j++)
                    {
                        if (i >= errors.Length || j >= weights.Length)
                            break;
                        for (int w = 0; w < errors[i].Length; w++)
                        {
                            outcomes[i] += weights[j][i] * errors[i][w];
                        }
                    }
                }
            }
            return outcomes;
        }

        //flip matrix on its side
        private float[][] transpose(float[][] weights)
        {
            float[][] newWeights = new float[weights[0].Length][];
            for (int i = 0; i < weights[0].Length; i++)
            {
                newWeights[i] = new float[weights.Length];
                for (int j = 0; j < weights.Length; j++)
                {
                    newWeights[i][j] = weights[j][i];
                }
            }
            return newWeights;
        }

        //unique getters and setters

        public float[] getOutput()
        {
            float[] outcomes = new float[control.outputs.nodes.Length];
            for (int i = 0; i < control.outputs.nodes.Length; i++)
            {
                outcomes[i] = control.outputs.nodes[i].output;
            }
            return outcomes;
        }

        public float getInput(int i)
        {
            return control.input.nodes[i].output;
        }

        public float[][][] getWeights()
        {
            float[][][] weights = new float[control.layers.Length][][];
            for (int i = 0; i < control.layers.Length; i++)
            {
                weights[i] = new float[control.layers[i].nodes.Length][];
                for (int j = 0; j < control.layers[i].nodes.Length; j++)
                {
                    weights[i][j] = control.layers[i].nodes[j].currentWeights;
                }
            }
            return weights;
        }

        public void setWeights(float[][][] weights)
        {
            param.weights = weights;
            for (int i = 0; i < control.layers.Length; i++)
            {
                for (int j = 0; j < control.layers[i].nodes.Length; j++)
                {
                    for (int w = 0; w < control.layers[i].nodes[j].currentWeights.Length; w++)
                        control.layers[i].nodes[j].currentWeights[w] = weights[i][j][w];
                }
            }
        }

        public Parameters getParameters()
        {
            return param;
        }
    }
}
