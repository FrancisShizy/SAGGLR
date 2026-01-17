# SAGGLR
Structure-Aware Compound-Protein Affinity Prediction via Graph Neural Network with Group Lasso Regularization
<img width="5315" height="2717" alt="Fig1_illustration_new" src="https://github.com/user-attachments/assets/a9c1512d-5d99-4b93-9bcb-9425d08cfd63" />

# Introduction
**Motivation:** Explainable artificial intelligence (XAI) approaches accelerate drug discovery by improving molecular representation learning, identifying key molecular structures, and rationalizing drug property prediction. However, building end-to-end explainable models for structure-activity relationship (SAR) modeling for compound property prediction faces many challenges, such as the limited number of compound-protein interaction activity data for specific protein targets, and plenty of subtle changes in molecular configuration sites significantly affecting molecular properties. Thus, optimally leveraging structural and property information and identifying key moieties related to compound–protein affinity for specific targets is essential. We proposed a framework by implementing graph neural networks (GNNs) to leverage property and structure information from pairs of molecules with activity cliffs targeting specific proteins to predict compound-protein affinity (i.e., half maximal inhibitory concentration, IC50) and explain property differences. To enhance model explainability, we trained GNNs with structure-aware loss functions using group lasso and sparse group lasso regularizations, which prune and highlight molecular subgraphs relevant to activity differences.

**Results:** We applied this framework to activity cliff data of molecules targeting three proto-oncogene tyrosine-protein kinase Src proteins (PDB IDs: 1O42, 2H8H, 4MXO). Integrating common and uncommon node information with sparse group lasso improves molecular property prediction for specific protein targets, as evidenced by lower root mean squared error (RMSE) and higher Pearson’s correlation coefficient (PCC). Applying regularizations also enhances feature attribution for GNNs by boosting graph-level global direction scores and improving atom-level coloring accuracy. These advances strengthen model interpretability in drug discovery pipelines, particularly in identifying critical molecular substructures in lead optimization.

# How to use
The data used in this paper is from this benchmark study, please see: https://github.com/josejimenezluna/xaibench_tf

```ruby
bash main.sh {cam | gradcam | gradinput | ig}
```
