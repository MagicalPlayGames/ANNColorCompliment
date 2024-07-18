using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Random = System.Random;
using UnityEditor;

[Serializable]
public class Wrapper<T>
{
    public T[] Items;
}
namespace MatchingColorPicker
{
    [Serializable]
    public struct testCase
    {
        public float[] firstColor;
        public float[] secondColor;
        public float[][] expectedThirdColors;

        public testCase(float[][][] colors)
        {
            firstColor = colors[0][0];
            secondColor = colors[1][0];
            expectedThirdColors = colors[2];
        }

        public void toString()
        {
            Console.WriteLine("First Color: R " + firstColor[0] + " G " + firstColor[1] + " B " + firstColor[2]);
            Console.WriteLine("Second Color: R " + secondColor[0] + " G " + secondColor[1] + " B " + secondColor[2]);
            Console.WriteLine("Third Color: R " + expectedThirdColors[0][0] + " G " + expectedThirdColors[0][1] + " B " + expectedThirdColors[0][2]);
        }
    }
    class TrainANN : MonoBehaviour
    {

        public static bool foundGood = false;
        public Parameters favoriteParameters;
        public int iterations;
        public int[] testCaseAmounts;
        public enum saveParams { None, All, BestOne };
        public enum runTypes {RunFromBeginning, GenerateBestOf20OfEachParameter, GenerateFavoriteWeights }
        private enum stepType { GenerateCases,GenerateBestANN, }
        public saveParams saveType;
        public runTypes runType;
        public bool training;
        private testCase[][] cases;
        public static string jsonPath = "ParameterCache";
        [Serializable]
        public struct LayerNodeLengths
        {
            public int[] lengths;
            public int this[int index]
            {
                get { return lengths[index]; }
                set { lengths[index] = value; }
            }

        }
        [SerializeField]
        public LayerNodeLengths[] trainingLayerNodeLengths;
        public void Awake()
        {
            training = this.gameObject.GetComponent<SelectColor>().training || training;
            if(runType==runTypes.RunFromBeginning)
            {
                training = true;
            }
            this.gameObject.GetComponent<SelectColor>().training = training;
        }

        public void Start()
        {
            if (training)
            {
                if (!PlayerPrefs.HasKey(jsonPath))
                {
                    PlayerPrefs.SetString(jsonPath, "");
                    PlayerPrefs.SetString(jsonPath+"_weights", "");
                }
                generateAllCases();
                GC.Collect();
                    switch (runType)
                    {
                        case runTypes.GenerateBestOf20OfEachParameter:
                            GC.Collect();
                            getBestOfParameters20();
                        break;
                        case runTypes.GenerateFavoriteWeights:
                            getWeights();
                            break;
                        case runTypes.RunFromBeginning:
                            while(favoriteParameters==null)
                                favoriteParameters = getBestOfParameters20();
                            GC.Collect();
                            getWeights();
                            runType = runTypes.GenerateFavoriteWeights;
                            break;
                    }
                GC.Collect();
                switch (runType)
                {
                    case runTypes.GenerateFavoriteWeights:
                        this.gameObject.GetComponent<SelectColor>().parameters = favoriteParameters;
                        this.gameObject.GetComponent<SelectColor>().testing();
                        break;
                }
            }
        }


        public void getWeights()
        {
            bool reapet = false;
            while (!foundGood)
            {
                float[][][] curWeights = generateWeights(reapet);
                if (curWeights != null)
                {
                    UnityEditor.AssetDatabase.Refresh();
                    string assetPath = UnityEditor.AssetDatabase.GetAssetPath(favoriteParameters);
                    favoriteParameters.setWeights(curWeights);
                    PlayerPrefs.SetString(jsonPath,JsonUtility.ToJson(favoriteParameters));
                    Wrapper<float> wrapper = new Wrapper<float>();
                    wrapper.Items = favoriteParameters.get1DWeights().ToArray();
                    Parameters p = ScriptableObject.CreateInstance<Parameters>();
                    if (assetPath.Length > 0)
                    {
                        PlayerPrefs.SetString(assetPath + "_weights", JsonUtility.ToJson(wrapper));
                        UnityEditor.AssetDatabase.DeleteAsset(assetPath);
                    }
                    else
                    {
                        assetPath = getNewAssetPath();
                        PlayerPrefs.SetString(assetPath + "_weights", JsonUtility.ToJson(wrapper));
                    }
                    JsonUtility.FromJsonOverwrite(PlayerPrefs.GetString(jsonPath), p);
                    float[] tempWeights;
                    JsonUtility.FromJsonOverwrite(PlayerPrefs.GetString(assetPath+"_weights"), wrapper);
                    tempWeights = wrapper.Items;
                    p.setWeights(tempWeights);
                    UnityEditor.AssetDatabase.CreateAsset(p, assetPath);
                    this.gameObject.GetComponent<SelectColor>().paramPath = assetPath;
                    UnityEditor.AssetDatabase.SaveAssets();
                    UnityEditor.AssetDatabase.Refresh();
                    favoriteParameters = p;
                }
                reapet = true;
            }
        }

        public float[][][] generateWeights(bool repeated)
        {
            Random r = new Random();
            BaseANN firstTry = new BaseANN();
            firstTry.setUpNetwork(favoriteParameters);
            if (repeated)
                firstTry.resetWeights();
            for (int index = 0; index < iterations; index++)
            {
                testCase[] testCases = cases[(int)r.Next(0, cases.Length)];
                int successes = 0;
                int total = 0;
                for (int casesIndex = 0; casesIndex < testCases.Length; casesIndex++)
                {
                    bool success = (firstTry.processData(true, new float[][] { new float[] { testCases[casesIndex].firstColor[0] }, new float[] { testCases[casesIndex].secondColor[0] } }, testCases[casesIndex].expectedThirdColors[0]));
                    if (!success)
                    {
                        firstTry.updateAllWeights();
                    }
                    else
                    {
                        successes++;
                    }
                    total++;
                    if (((float)successes / (float)total < .1f || (float)successes / (float)total > .8f) && casesIndex > iterations * .25f)
                    {
                        break;
                    }
                }

                if ((float)successes / (float)total > .75f)
                {
                    foundGood = true;
                    return firstTry.getWeights();
                }
            }
            return null;
        }

        public Parameters getBestOfParameters20()
        {
            Parameters prevBestAI, curBestAI;
            prevBestAI = null;
            for (int i = 0; i < 20; i++)
            {
                curBestAI = generateBestANN();
                if (curBestAI != null)
                {
                    if (prevBestAI == null)
                    {
                        prevBestAI = curBestAI;
                    }
                    else if (prevBestAI.trainingSuccessRate < curBestAI.trainingSuccessRate)
                    {
                        prevBestAI = curBestAI;
                    }
                }
            }
            if (prevBestAI != null && saveType == saveParams.BestOne)
            {
                PlayerPrefs.SetString(jsonPath, JsonUtility.ToJson(prevBestAI));
                Wrapper<float> wrapper = new Wrapper<float>();
                wrapper.Items = prevBestAI.get1DWeights().ToArray();
                PlayerPrefs.SetString(jsonPath+"_weights", JsonUtility.ToJson(wrapper));
                saveParameters();
            }
            return prevBestAI;
        }

        public void generateAllCases()
        {
            cases = new testCase[testCaseAmounts[0]][];
            for (int i = 0; i < cases.Length; i++)
            {
                cases[i] = generateTestCases(testCaseAmounts[1]);
            }

        }

        public Parameters generateBestANN()
        {
            Random r = new Random();

            for (byte j = 0; j < 3; j++)
                for (byte paramIndex = 0; paramIndex < trainingLayerNodeLengths.Length; paramIndex++)
                {
                    for (float w = 0.01f; w < .4f; w += 0.02f)
                    {
                        BaseANN firstTry = new BaseANN();
                        int[] curNodes = trainingLayerNodeLengths[paramIndex].lengths;
                        firstTry.setUpNetwork((byte)curNodes.Length, curNodes, w, j, 1);
                        for (int index = 0; index < iterations; index++)
                        {
                            testCase[] testCases = cases[(int)r.Next(0, cases.Length)];
                            int successes = 0;
                            int total = 0;
                            for (int casesIndex = 0; casesIndex < testCases.Length; casesIndex++)
                            {
                                bool success = (firstTry.processData(true, new float[][] { new float[] { testCases[casesIndex].firstColor[0], testCases[casesIndex].secondColor[0] } }, testCases[casesIndex].expectedThirdColors[0]));
                                if (!success)
                                {
                                    firstTry.updateAllWeights();
                                }
                                else
                                {
                                    successes++;
                                }
                                total++;
                                if ((float)successes / (float)total < .1f && casesIndex > iterations * .5f)
                                {
                                    break;
                                }
                                else if ((float)successes / (float)total > .7f && casesIndex > iterations * .5f)
                                {
                                    Parameters pars = firstTry.getParameters();
                                    pars.setSuccessRate((float)successes / (float)total);
                                    pars.setWeights(firstTry.getWeights());
                                    if (saveType == saveParams.All)
                                    {
                                        PlayerPrefs.SetString(jsonPath, JsonUtility.ToJson(pars));
                                        Wrapper<float> wrapper = new Wrapper<float>();
                                        wrapper.Items = pars.get1DWeights().ToArray();
                                        PlayerPrefs.SetString(jsonPath + "_weights", JsonUtility.ToJson(wrapper));
                                        saveParameters();
                                    }
                                    return pars;
                                }
                            }

                        }
                    }
                }
            return null;
        }


        public testCase[] generateTestCases(int testCases)
        {
            testCase[] cases = new testCase[testCases];
            for (int i = 0; i < testCases; i++)
            {
                cases[i] = new testCase(generateCase());
            }
            return cases;
        }

        public float[][][] generateCase()
        {
            int maxTries = 25;
            float[] color1 = new float[3];
            float[] color2 = new float[3];
            float[] color3 = new float[3];
            color1 = generateColor();
            color2 = generateColor();
            color3 = generateColor();
            while (!(checkColor(color1, color3) && checkColor(color2, color3)) && --maxTries>=0)
            {
                if (checkColor(color1, color3))
                {
                    color2 = generateColor();
                }
                else if (checkColor(color2, color3))
                {
                    color1 = generateColor(); ;
                }
                else
                {
                    color3 = generateColor();
                }
            }
            if(maxTries<0)
                return generateCase();
            return new float[][][] { new float[][] { color1 }, new float[][] { color2 }, new float[][] { color3 } };
        }

        public float[] generateColor()
        {
            Random r = new Random();
            return new float[] { (float)r.NextDouble(), (float)r.NextDouble(), (float)r.NextDouble()};
        }

        public bool checkColor(float[] c1, float[] c2)
        {
            float sum = 0;
            for (int i = 0; i < 1; i++)
            {
                sum += Math.Abs(c2[i] - c1[i]);
            }
            return (sum / 1.0f) > .5f;
        }


        public static void saveParameters()
        {
            Parameters param = ScriptableObject.CreateInstance<Parameters>();
            JsonUtility.FromJsonOverwrite(PlayerPrefs.GetString(jsonPath),param);
            if (!UnityEditor.AssetDatabase.IsValidFolder("Assets/Scripts/ANNParameters/" + DateTime.Now.ToString("dd-MM-yy")))
            {
                UnityEditor.AssetDatabase.CreateFolder("Assets/Scripts/ANNParameters", DateTime.Now.ToString("dd-MM-yy"));
                UnityEditor.AssetDatabase.Refresh(); 
            }
            
            UnityEditor.AssetDatabase.CreateAsset(param, getNewAssetPath());
            UnityEditor.AssetDatabase.SaveAssets();
            UnityEditor.AssetDatabase.Refresh();
            PlayerPrefs.SetString(UnityEditor.AssetDatabase.GetAssetPath(param) + "_weights",PlayerPrefs.GetString(jsonPath+"_weights"));
        }

        public static string getNewAssetPath()
        {
            return "Assets/Scripts/ANNParameters/" + DateTime.Now.ToString("dd-MM-yy") + "/Parameter_Card_" + DateTime.Now.ToString("hh-mm-ss") + ".asset";
        }
    }
}
