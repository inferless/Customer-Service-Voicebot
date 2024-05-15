# Customer-Service-Voicebot
- Welcome to an engaging tutorial designed to walk you through creating a customer support voicebot where users can voice their queries and receive solutions. You'll learn to integrate speech recognition, large language, and text-to-speech models to develop a responsive and efficient voice-based customer support system.
## Architecture
<img width="1336" alt="image" src="https://github.com/inferless/Customer-Service-Voicebot/assets/150957746/72ff3c30-f543-4845-a14e-7efa0712ca33">

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

  ---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
    def infer(self, inputs):
        audio_data = inputs["audio_base64"]
        
        #Convert the audio from base64 to .mp3
        self.base64_to_mp3(audio_data, self.audio_file)

        # Transcribe audio file
        segments, info = self.model_whisper.transcribe(self.audio_file, beam_size=5)
        user_text = ''.join([segment.text for segment in segments])

        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": "You are Customer Support Assistant."},
            {"role": "user", "content": user_text}
        ]

        # Generate input for the LLM
        gen_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)

        # Query the vector store
        response = self.query_engine.query(gen_input)

        # Synthesize response to audio
        byte_stream = io.BytesIO()
        with wave.open(byte_stream, "wb") as wav_file:
            # Set WAV file parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(22050)  # Sample rate

            # Synthesize the speech and write to byte stream
            self.voice.synthesize(response.response, wav_file)

        # Get the byte stream's content
        audio_bytes = byte_stream.getvalue()

        # Encode audio bytes to Base64 string
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {"generated_audio_base64": audio_base64,
               "question":user_text,
               "answer":response.response}
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
