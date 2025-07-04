# YouTube Product Review Sentiment Analyzer

## Introduction

Our project analyzes comment sentiment on Youtube product review videos. To achieve this, we fine-tuned a pre-trained **BERT** model (`"bert-base-uncased"`) with 1,185 manually labeled entries. 

To download our fine-tuned model, click [here](https://drive.google.com/file/d/1P52GL9VLMxN9SAsMYR7SdOkV9b7TUQ4N/view?usp=sharing). 

We also built a user-friendly web interface, developed with Streamlit, enabling users to input a video URL to receive a summarized sentiment analysis for both the product and the video.

This work represents our final course project for ISE 540 (Text Analytics), completed by Zijin Qin, Aria Xu, Abhijeet Nijjar, and Junhao Liang.


## How to Deploy Application/Project

### Clone the Project Files

Select a local directory and clone this project: 

```
git init
git clone https://github.com/Jziumo/ISE-540-Final-Project.git
```

Create a new virtual environment venv in current directory. (It's recommended to use python 3.10 to avoid version incompatibility.)

```
python -m venv ./venv/
```

Activate the virtual environment. If you are using **Windows**, run this: 

```
.\venv\Scripts\Activate.ps1
```

If you are using **Linux**, run this: 

```
source ./venv/bin/activate
```

Then you should see `(venv)` on the left side of your command line. (Not `(base)`!)

### Download the Model Files

To download our fine-tuned model, click [here](https://drive.google.com/file/d/1P52GL9VLMxN9SAsMYR7SdOkV9b7TUQ4N/view?usp=sharing). 

Then, unzip and put the files in the root directory of the project like this: 

```
your_project_root/
├── models/
│   ├── sentiment_for_product_stable/
│   └── sentiment_for_video_stable/  
```

### Obtain Key for Youtube Data API

Since currently we are using **Youtube Data API v3** to obtain video comments, the application needs a key for the API to work.

Obtain a key for Youtube Data API following: 
- Go to [Google Developers Console](https://console.developers.google.com/)
- Search `YouTube Data API v3`
- Choose **'Public API'** to obtain a key

Put your key in `config.toml` like this: 

```
[api_keys]
youtube = "YOUR KEY"
```

### Deploy the Web Application

Install the libraries used in these scripts. 

```
pip install -r requirements_app.txt
```

Then start the web application by running: 

```
start
```

If it doesn't work, just directly run `main.py`:

```
python -m src.main
``` 

### Train the Model

If you want to train the model on your own data, install these libraries: 

```
pip install -r requirements_train.txt
```

Make sure your CUDA environment is compatible with CUDA 12.1. 

Then modify and run `model.py`. The model files will be saved in `./models`: 

```
your_project_root/
├── models/
│   ├── sentiment_for_product_current/
│   └── sentiment_for_video_current/  
```


## Other Links: 

[Tutorial: Label the Data](./doc/label_data_tutorial.md)