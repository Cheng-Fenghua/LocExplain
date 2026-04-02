# LocExplain
This is the supplemntary material of submission: "LocExplain: A Multimodal Dataset and Benchmark for Explainable Street-View Geo-Localization". We provide the whole dataset and instructions on how to access it in this page. The repository structure is as below:

LocExplain/
├── readme.md
├── LocExplain_explanations.json
├── LocExplain_train_test_split.json
├── A LocExplain Example - Panoramas/
│   ├── ...(images)
├── LocKnowledge_knowledge_text.json
├── A LocKnowledge Example - China/
│   ├── ...(images)
├── Results/
│   ├── deepseekvl2_results.json
│   ...
│   ├── SightSense_results.json
├── SightSense/
│   ├── dataset.py
│   ...
│   ├── utils.py

The whole LocExplain (paronamas) can be accessed via link: 
The whole LocKnowledge can be accessed via link: 

LocExplain_explanations.json - provides ground truth explanations
LocExplain_train_test_split.json - provides ground truth explanations
A LocExplain Example - provides an example of panoramas in LocExplain (full dataset can be accessed via link above)
LocKnowledge_knowledge_text.json - provides knowledge text
A LocKnowledge Example - provides an example of image set in LocKnowledge (full knowledge set can be accessed via link above)
Results - provides generated results by SightSense and baseline models
SightSense - provides all code of SightSense
