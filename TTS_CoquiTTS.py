import tempfile ,os
from TTS.config import load_config
import gradio as gr

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

MODEL_NAMES=[
    "vits male1 (best)",
    "vits female (best)",
    "vits-male",
    "vits female1",
    "glowtts-male",
    "glowtts-female",
    "female tacotron2"
]
MAX_TXT_LEN = 800
ROOT = r"D:\Git_repos\DeepNoiseReducer\models\PersianTTS"
os.chdir(ROOT)

model_path = os.getcwd() + "/best_model.pth"
config_path = os.getcwd() + "/config.json"


from TTS.utils.download import download_url
modelInfo=[
    ["vits-male","best_model_65633.pth","config-0.json","https://huggingface.co/Kamtera/persian-tts-male-vits/resolve/main/"],
    # ["vits female (best)","checkpoint_48000.pth","config-2.json","https://huggingface.co/Kamtera/persian-tts-female-vits/resolve/main/"],
    # ["glowtts-male","best_model_77797.pth","config-1.json","https://huggingface.co/Kamtera/persian-tts-male-glow_tts/resolve/main/"],
    # ["glowtts-female","best_model.pth","config.json","https://huggingface.co/Kamtera/persian-tts-female-glow_tts/resolve/main/"],
    # ["vits male1 (best)","checkpoint_88000.pth","config.json","https://huggingface.co/Kamtera/persian-tts-male1-vits/resolve/main/"],
    # ["vits female1","checkpoint_50000.pth","config.json","https://huggingface.co/Kamtera/persian-tts-female1-vits/resolve/main/"],
    # ["female tacotron2","checkpoint_313000.pth","config-2.json","https://huggingface.co/Kamtera/persian-tts-female-tacotron2/resolve/main/"]
]

# for d in modelInfo:

#     directory=d[0]
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     print("|> Downloading: ",directory)
#     download_url(
#         d[3]+d[1],directory,"best_model.pth"
#     )
#     download_url(
#         d[3]+d[2],directory,"config.json"
#     )
def tts(text: str,model_name: str):
    if len(text) > MAX_TXT_LEN:
        text = text[:MAX_TXT_LEN]
        print(f"Input text was cutoff since it went over the {MAX_TXT_LEN} character limit.")
    print(text)

    
    # synthesize
    synthesizer = Synthesizer(
        model_name+"/best_model.pth", model_name+"/config.json"
    )
    if synthesizer is None:
        raise NameError("model not found")
    wavs = synthesizer.tts(text)
    # return output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        synthesizer.save_wav(wavs, fp)
        return fp.name


description="""
This is a demo of persian text to speech model.
**Github : https://github.com/karim23657/Persian-tts-coqui  **
Models can be found here:  <br>
|Model|Dataset|
|----|------|
|[vits female (best)](https://huggingface.co/Kamtera/persian-tts-female-vits)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[vits male1 (best)](https://huggingface.co/Kamtera/persian-tts-male1-vits)|[persian-tts-dataset-male](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-male)|
|[vits female1](https://huggingface.co/Kamtera/persian-tts-female1-vits)|[ParsiGoo](https://github.com/karim23657/ParsiGoo)|
|[vits male](https://huggingface.co/Kamtera/persian-tts-male-vits)|[persian-tts-dataset](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset)|
|[glowtts female](https://huggingface.co/Kamtera/persian-tts-female-glow_tts)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[glowtts male](https://huggingface.co/Kamtera/persian-tts-male-glow_tts)|[persian-tts-dataset](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset)|
|[tacotron2 female](https://huggingface.co/Kamtera/persian-tts-female-tacotron2)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
"""
article= ""
examples=[
    ["و خداوند شما را با ارسال روح در جسم زندگانی و حیات بخشید","vits-male"],
    ["تاجر تو چه تجارت می کنی ، تو را چه که چه تجارت می کنم؟","vits female (best)"],
    ["شیش سیخ جیگر سیخی شیش هزار","vits female (best)"],
    ["سه شیشه شیر ، سه سیر سرشیر","vits female (best)"],
    ["دزدی دزدید ز بز دزدی بزی ، عجب دزدی که دزدید ز بز دزدی بزی","vits male1 (best)"],
    ["مثنوی یکی از قالب های شعری است ک هر بیت قافیه ی جداگانه دارد","vits female1"],
    ["در گلو ماند خس او سالها، چیست آن خس مهر جاه و مالها","vits male1 (best)"],
]
iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.Textbox(
            label="Text",
            value="زندگی فقط یک بار است؛ از آن به خوبی استفاده کن",
        ),
        gr.Radio(
            label="Pick a TTS Model ",
            choices=MODEL_NAMES,
            value="vits-female",
        ),
    ],
    outputs=gr.Audio(label="Output",type='filepath'),
    examples=examples,
    title="🗣️ Persian tts 🗣️",
    description=description,
    article=article,
    live=False
)
iface.launch(share=True)
