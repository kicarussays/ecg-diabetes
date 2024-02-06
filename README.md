# Deep learning-based long-term risk evaluation of incident type 2 diabetes using electrocardiogram in a non-diabetic population: a retrospective, multicentre study (DOI:https://doi.org/10.1016/j.eclinm.2024.102445)

## Background. 
Diabetes is a major public health concern. This study aimed to evaluate the long-term risk of incident type 2 diabetes in a non-diabetic population using a deep learning model (DLM) detecting prevalent type 2 diabetes using electrocardiogram (ECG). 

## Methods. 
Participants who underwent health checkups at two hospitals between January 2001 and December 2022 were included in the study. Type 2 diabetes was defined as glucose levels ≥ 126 mg/dL or glycated haemoglobin (HbA1c) levels ≥ 6·5%. The one-dimensional ResNet-based model detected prevalent type 2 diabetes, and the Guided Grad-CAM was used to localise important regions. For long-term risk evaluation, we assumed that individuals without diabetes with diabetic ECG had a higher risk of incident type 2 diabetes than those with non-diabetic ECG. We divided the non-diabetic group into well-classified (with non-diabetic ECG) and misclassified (with diabetic ECG) groups and performed a Cox proportional hazard model, considering the occurrence of type 2 diabetes more than six months after the visit.

## Findings. 
A total of 190581 individuals were included in the study. The areas under the receiver operating characteristic curve for prevalent type 2 diabetes detection were 0·816 (0·807–0·825) and 0·762 (0·754–0·770) for the internal and external validations, respectively. The model primarily focused on the QRS duration and, occasionally, P or T waves. The non-diabetic group with diabetic ECG exhibited an increased risk of incident type 2 diabetes compared with the non-diabetic group with non-diabetic ECG, with hazard ratios of 2·15 (1·82–2·53) and 1·92 (1·74–2·11) for internal and external validation, respectively.

## Interpretation. 
This study focused on risk evaluation of incident type 2 diabetes among individuals without diabetes using the DLM. In the non-diabetic group, those whose ECG was classified as diabetic by the DLM were at a higher risk of incident type 2 diabetes than those whose ECG was not.
