# LocExplain
This is the supplemntary material of submission: "LocExplain: A Multimodal Dataset and Benchmark for Explainable Street-View Geo-Localization". We provide the whole dataset and instructions on how to access it in this page. The repository structure is as below:

LocExplain/<br>
├── readme.md<br>
├── LocExplain_explanations.json<br>
├── LocExplain_train_test_split.json<br>
├── A LocExplain Example - Panoramas/<br>
│   ├── ...(images)<br>
├── LocKnowledge_knowledge_text.json<br>
├── A LocKnowledge Example - China/<br>
│   ├── ...(images)<br>
├── Results/<br>
│   ├── deepseekvl2_results.json<br>
│   ...<br>
│   ├── SightSense_results.json<br>
├── SightSense/<br>
│   ├── dataset.py<br>
│   ...<br>
│   ├── utils.py<br>

The whole LocExplain (paronamas) can be accessed via link: https://drive.google.com/file/d/1eVWh9Tywohw-wz9VgNRr3JST42nhzmky/view?usp=sharing<br>
The whole LocKnowledge can be accessed via link: https://drive.google.com/file/d/19A6tCIFUfW6YbAKY-bnEqFxpExGPsghi/view?usp=sharing

LocExplain_explanations.json - provides ground truth explanations<br>
LocExplain_train_test_split.json - provides ground truth explanations<br>
A LocExplain Example - provides an example of panoramas in LocExplain (full dataset can be accessed via link above)<br>
LocKnowledge_knowledge_text.json - provides knowledge text<br>
A LocKnowledge Example - provides an example of image set in LocKnowledge (full knowledge set can be accessed via link above)<br>
Results - provides generated results by SightSense and baseline models<br>
SightSense - provides all code of SightSense
