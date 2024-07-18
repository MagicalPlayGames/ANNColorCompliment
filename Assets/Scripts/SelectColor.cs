using MatchingColorPicker;
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class SelectColor : MonoBehaviour
{
    public enum colorValue { Red, Green, Blue };
    public struct colorSlider
    {
        public float R;
        public float G;
        public float B;

        public colorSlider(float R, float G, float B)
        {
            this.R = R;
            this.G = G;
            this.B = B;
        }

        public Color getColor()
        {
            return new Color(R, G, B, 1);
        }

        public float[] getFloats()
        {
            return new float[] { R, G, B };
        }

        public void setVar(float value, colorValue index)
        {
            switch (index)
            {
                case colorValue.Red:
                    R = value;
                    break;
                case colorValue.Green:
                    G = value;
                    break;
                case colorValue.Blue:
                    B = value;
                    break;
            }

        }

        public string getHex()
        {
            return "#" + ColorUtility.ToHtmlStringRGB(new Color(R, G, B));
        }
    }

    public colorSlider color1;
    public colorSlider color2;
    public Slider[] sliders;
    public Color[] colors;
    private BaseANN brain;
    public Material mat;
    public Text[] colorText;
    public Parameters parameters;
    public bool training;
    public string paramPath;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public void Start()
    {
        if (!training)
        {
            paramPath = UnityEditor.AssetDatabase.GetAssetPath(parameters);
            testing();
        }
    }

    public void testing()
    {
        color1 = new colorSlider(sliders[0].value, sliders[1].value, sliders[2].value);
        color2 = new colorSlider(sliders[3].value, sliders[4].value, sliders[5].value);
        colors[0] = color1.getColor();
        colors[1] = color2.getColor();
        setColor(0);
        setColor(1);
        brain = new BaseANN();
        setUpNetwork();
        thirdColor();
    }

    private void OnApplicationQuit()
    {
        brain = null;
        GC.Collect();
    }

    public void sliderValueChanged(int sliderIndex)
    {
        colorValue index = colorValue.Red;
        index += (sliderIndex % 3);
        if (sliderIndex < 3)
        {
            color1.setVar(sliders[sliderIndex].value, index);
            colors[0] = color1.getColor();
            setColor(0);
        }
        else
            color2.setVar(sliders[sliderIndex].value, index);
        colors[1] = color2.getColor();
        setColor(1);
    }
    private void setUpNetwork()
    {

        Wrapper<float> wrapper = new Wrapper<float>();
        JsonUtility.FromJsonOverwrite(PlayerPrefs.GetString(paramPath + "_weights"), wrapper);
        parameters.setWeights(wrapper.Items);
        brain.setUpNetwork(parameters);
    }

    public void thirdColor()
    {
        brain.processData(false, new float[][] { new float[] { colors[0].r}, new float[] { colors[1].r} });
        float[] colorFloats = new float[3];
        colorFloats[0] = brain.getOutput()[0]*(colors[0].r+colors[1].r)/2.0f;
        brain.processData(false, new float[][] { new float[] { colors[0].g }, new float[] { colors[1].g } });
        colorFloats[1] = brain.getOutput()[0] * (colors[0].g + colors[1].g) / 2.0f;
        brain.processData(false, new float[][] { new float[] { colors[0].b }, new float[] { colors[1].b } });
        colorFloats[2] = brain.getOutput()[0] * (colors[0].b + colors[1].b) / 2.0f;
        colors[2] = new Color(colorFloats[0],colorFloats[1],colorFloats[2]);
        setColor(2);
    }

    private void setColor(int index)
    {
        mat.SetColor("_Color" + (index + 1), colors[index]);
        switch (index)
        {
            case 0:
                colorText[0].text = color1.getHex();
                colorText[0].color = color1.getColor();
                break;
            case 1:
                colorText[1].text = color2.getHex();
                colorText[1].color = color2.getColor();
                break;
            case 2:
                colorText[2].text = "#"+ColorUtility.ToHtmlStringRGB(colors[2]);
                colorText[2].color = colors[2];
                break;
        }
    }
}
