# **GRAAFE: GRaph Anomaly Anticipation Framework for Exascale HPC systems**

### Authors: BLIND for review



![Screenshot from 2023-04-13 11-08-35](https://user-images.githubusercontent.com/13011878/231729695-fb6b5e87-b932-49df-859f-b87156e1b994.png)

This repository contains two main folders, including codes and other materials necessary to reproduce the paper's results. The first folder, "offline," focuses on training the GNN model. The second folder, "online," covers the deployment evaluation of monitoring and MLOps frameworks in HPC.

# Monitoring System (ExaMon)

Before delving into the offline and online approaches for training and deploying the GNN model, it's important to clarify that the details of the monitoring system are beyond the scope of this repository. Instead, this repository serves as a client of the monitoring system and utilizes the examon client library. The `examon` folder contains the necessary examon client libraries to extract data from the monitoring system. For more information about the monitoring system, please refer to [this repository](https://github.com/EEESlab/examon/tree/develop/docker) and [this paper](https://doi.org/10.1145/3339186.3339215).

# Offline

The offline folder contains all scripts necessary to validate the Offline experimental part of the paper.
This includes training the GNN models, comparison against the per-node baselines, and the analysis of the results 
(calculation of the AUC and producing the visualization).

# Online

To reproduce the results of the Deployment Evaluation of Monitoring and MLOps Frameworks in HPC, you need four main requirements.
First, you need a datacenter monitoring system that collects online datacenter metrics such as power and CPU frequency.
Second, you need well-trained GNN models that are trained on historical data.
Third, you need a cloud system with Kubernetes and Kubeflow installed. For small-scale tests, you can even use your laptop for the last requirement.
Finally, a set of Python scripts connect to the monitoring system, extract and preprocess data, perform inference (anomaly prediction), and send/publish the anomaly prediction results back to the monitoring system. Additionally, there are several other YAML and text files required for the other steps.  
The following provides more detailed information about these requirements.

## Kubeflow

Kubeflow is an open-source machine-learning tool that is built on top of Kubernetes. As such, it requires Kubernetes to be installed. Here is a useful [link](https://charmed-kubeflow.io/docs/get-started-with-charmed-kubeflow) to a tutorial that provides step-by-step instructions on how to install Microk8s - a lightweight version of Kubernetes that can even be installed on your laptop. Additionally, the tutorial provides guidance on how to install and use Kubeflow. Microk8s is a great option for those who are new to Kubernetes and want to get started with Kubeflow quickly and easily. Once you have installed Microk8s, you can follow the tutorial to install and use Kubeflow. With Kubeflow, you can easily develop and deploy machine learning models at scale.

## Monitoring System Clinet (ExaMon Clinet)
The examon folder contains the examon client libraries required to extract data from the Examon monitoring system. In the data_extraction.py file, we used the Examon client library to extract monitoring metrics that are necessary for the GNN model. We called the function defined in data_extraction.py from main.py.
For more information about the monitoring system, refer to [this repository](https://github.com/EEESlab/examon/tree/develop/docker) and [this paper](https://doi.org/10.1145/3339186.3339215).




## Models

The `models` folder contains trained GNN models, including their weights and parameters, for the CINECA Marconi100 HPC system..

## Python Scripts

The python `*.py` files in general contain scripts provided that provide necessary scripts for doing anomaly prediction. 

`data_extraction.py`: This Python script uses the libraries in the `examon` folder to extract monitoring data of the compute nodes from the monitoring system. The `node_names` file contains the list of compute nodes for the CINECA Marconi100 HPC system, and the `metrics` file contains the list of metrics from which we need to extract data. To connect and extract data from ExaMon, you need an account on this monitoring system.

`preprocessing.py` : This Python script provides some functions that are needed to convert the raw data extracted from ExaMon to the data format that is useful for the GNN models. With function `agg_df_avg_min_max_std(df: Pandas.DataFrame)` computes the new features (std, min, max, mean) for the `metrics` and then the function `convert_to_graph_data(df: Pandas.DataFrame)` converts the Pandas.DataFrame to the graph using `torch_geometric.data` class of torch which is a format the GNN model receives input data.    

`inference.py`: This Python file contains scripts that create a GNN model with PyTorch.  

`publishing_results.py`: This Python file contains scripts that sned (publish) the results of predictions to the monitoring system using the MQTT protocol.   

`logging_module.py`: This Python file contains scripts for creating a logging system for necessary parameters.  

`main.py`: This Python file contains all the scripts needed to use the functions defined in other Python files for (I) data extraction, (II) preprocessing, (III) inference, and finally (IV) sending the prediction results to the monitoring system. The `main.py` requires several arguments to run. For example: `python3 main.py -euser 'XYZ' -epwd 'XYZ' -r 'r256' -bs '192.168.0.35' -ir 0` Below is a list of the arguments required for `main.py`, along with brief help and default values. The arguments can be divided into three sets: those necessary for the monitoring system for data extraction, those necessary for the monitoring system for publishing the results, and those necessary for the inference and GNN model parameters.




```bash
#### arguments necessary for the inference and GNN model parameters.

-ir, --inference_rate, type=int, help=This shows the inference rate in seconds.,default=0

-r, --rack_name, type=str, help=Rack Name, all mesans all racks of the M100 one by one in serial approach. ,default='r256'

-ph, --prediction_horizon, type=int, help=Prediction Horizon. ,default=24

#### arguments necessary for the monitoring system for data extraction

-es, --examon_server, type=str, help=KAIROSDB_SERVER = "examon.cineca.it", default="examon.cineca.it"

-ep, --examon_port, type=str, help=KAIROSDB_PORT = "3000",default="3000"

-euser, --examon_user, type=str, help=Examon Username

-epwd, --examon_pwd, type=str, help=Examon Password

#### arguments necessary for the monitoring system for publishing the inference results 

-bs, --broker_address, type=str, help=broker_address = "192.168.0.35" or broker_address = "examon.cineca.it", default="192.168.0.35"

-bp, --broker_port, type=int, help=broker_port = 1883, default=1883

```



## Dockerfile

A `Dockerfile` is a recipe for creating a Docker image.

## Other Files

The `requirements.txt` file lists the software package requirements for Python scripts. `node_names` is a list of names of compute nodes, and `metrics` is a list of metrics from which we need to extract data from the monitoring system.

## GitHub Actions

The `github_action_config.yaml` file provides configuration scripts for automating the build and push of a Docker image to Docker Hub using [GitHub Actions](https://docs.github.com/en/actions). Here's a quick summary of the steps involved:

1. In your repository, go to Settings → Secrets and variables → Actions → New repository secret, and create two new secrets for your Docker Hub username and password.
2. In your GitHub repository, go to Actions → New workflow → Suggested for this repository, or set up the workflow yourself.
3. Copy and paste the scripts provided in `github_action_config.yaml` into the workflow file. Make sure to use the correct secret names for the username and password. In "docker build" and "docker push", specify the name of the Docker image you prefer. In this case, we used "gnnexamon".
4. Push the commit to trigger the workflow.

For more detailed instructions, refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions).

**Note:** For reproducing all steps in the way that we did, you need to have a [docker hub](https://www.notion.so/Docker-d5fcbd6532e64a80af7125a9d6bc5912) and [GitHub](https://github.com/) accounts. But these are not mandatory since you can just download the [docker image](https://hub.docker.com/repository/docker/kazemi/gnnexamon/general) that we create, and you can find it in [this link.](https://hub.docker.com/repository/docker/kazemi/gnnexamon/general) 







## Kubeflow Pipeline

The `gnn_pipelines.ipynb` is a Jupyter notebook that contains scripts for creating and running Kubeflow pipelines. Kubeflow Pipelines (kfp) is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers. [Here in this link, you can find more information about the kfp.](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/)

The Kubeflow Pipelines SDK provides a set of Python packages that you can use to specify and run your machine learning (ML) workflows. A pipeline is a description of an ML workflow, including all of the components that make up the steps in the workflow and how the components interact with each other.

Like the following scripts, to create the Kubeflow pipeline, first, we create a component of the pipeline using the component package and the class `components.load_component_from_text()`, which loads components from text and creates a task factory function. We then create components from the Docker image by pulling the image from `docker.io/kazemi/gnnexamon`. Next, we execute `main.py` from the directory `/hpc_gnn_mlops/` inside the Docker container, defining the correct values for arguments such as the examon username and password, the name of the rack and broker server, and the inference rate.   

```python
gnn= components.load_component_from_text("""
name: Online Prediction
description: A pipeline that performs anomaly prediction on a Tier-0 supercomputer.

implementation:
  container:
    image: kazemi/gnnexamon
    command: [
    "/bin/sh",
    "-c",
    "cd /hpc_gnn_mlops/ && python3 main.py -euser 'XYZ' -epwd 'XYZ' -r 'r256'  -bs '192.168.0.35' -ir 0"
    ]
""")
```



After creating this component, we should define the pipeline. 

```python
# Define the pipeline
@dsl.pipeline(
   name=PIPELINE_NAME,
   description=PIPELINE_DESCRIPTION
)

def gnn_pipeline():
    gnn_obj = gnn()
```

Finally, we can create and run a Kubeflow run - experiment.

```python
# Specify pipeline argument values
arguments = {} 
kfp.Client().create_run_from_pipeline_func(gnn_pipeline, arguments=arguments, experiment_name=EXPERIMENT_NAME)    
```

The `gnn_pipelines.ipynb` notebook contains six different experiments that we conducted on paper.

## Some additional experimental results
In this section, some of the additional experimental results are collected. These results are supplementary to the main paper and help add more detail. They were cut from the final version of the paper to comply with the journal's limit of 8 pages.

### Probability calibration of the GNN anomaly prediction model
The GNN anomaly anticipation model's main output is the anomaly's probability in the next time window. For this reason, the calibration of the probability predictions of the GNN classifier is essential. Calibration refers to ensuring that the predicted probabilities from a classifier align with the actual anomaly frequencies in a test set.

The Brier score, a widely used measure for evaluating the calibration of probability predictions in binary classification cases, assesses the mean squared difference between predicted probabilities and actual outcomes. This metric yields values ranging from 0 to 1, where perfect calibration is indicated by a score of zero. It should be noted that the Brier score does not evaluate classification performance directly but instead focuses solely on the calibration aspect.

The results of the Brier score calculation on the test set are collected in table bellow. Here we see that the GNN achieves (for all future windows) a probability score of fewer than 0.1. According to expectation, the calibration score rises with larger future windows as the approach attempts to make predictions on a more difficult problem. For future windows 4 and 6 (one hour and one and a half hours) Brier score is below 0.01, indicating an extremely well-calibrated classifier.

The results from the calibration evaluation mean that no additional calibration adjustment needs to be performed on the probability predictions of the GNN classifier. The probability outputs of the GNN model can be communicated directly to the system administrators as discussed in section V.3 of the main paper.


| FW          | 4     | 6     | 12    | 24    | 32    | 32    | 64    | 96    | 192   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Brier Score | 0.008 | 0.009 | 0.012 | 0.017 | 0.020 | 0.031 | 0.040 | 0.063 | 0.083 |


### Computational resources requirements




The table reports the computation resource requirements and deployment overhead for the anomaly prediction pipeline on the
 ExaMon monitoring system. The table is divided into different components, providing a deep insight into the design and architectural
 view of the monitoring system. We provided a baseline that shows the normal load and traffic of the system without running any anomaly detection pipeline. This result is a basis for estimating the overhead of deploying DNN models and benefiting future per-exascale design.

Different setups impact resource utilization differently; in many cases, the impact is almost negligible, while in a few others, it is more marked. The resource usage of some parts of the framework mainly stayed the same during the pipeline. This indicates that running pipelines has no significant effect on these parts. 

Using multiple pods in a continuous approach increases the CPU load of the monitoring system by 2.5 times. Conversely, using the one discrete pod has a minimal impact on memory RAM usage (practically no difference) and a very limited effect on CPU resources, namely a 10% increase (from 3.08 to 3.42, as shown in the table). 

**The results** collected in this section **demonstrate that the GNN model deployment, as part of the GRAAFE framework, has negligible computational overhead compared to only the monitoring system**. This makes the GNN anomaly detection and the **GRAAFE framework suitable for deployment in most High-performance computing and data center environments**.

| **Configuration Name: Proxy**|            **#vcores**                            |       **Mem[GB]** |     **Net in[Kb/s]**   |          **Net out[Kb/s]**                  |
|--------------------------------------|:--------------------:|:-----------:|:----------------:|:-----------------:|
| Baseline                             |                 0.18 |        2.64 |              838 |               832 |
| One Rack, One Pod, Continuous        |                 0.64 |         2.6 |              960 |               956 |
| One Rack, One Pod, Discrete          |                 0.19 |        2.59 |              899 |               895 |
| All Racks, Multiple Pods, Continuous |                 1.98 |        2.64 |             1282 |              1289 |
| All Racks, Multiple Pods, Discrete   |                 0.34 |        2.61 |              937 |               935 |
| All Racks, One Pod, Continuous       |                 0.32 |        2.61 |              990 |               987 |
| All Racks, One Pod, Discrete         |                 0.19 |        2.61 |              924 |               922 |
|  **Configuration Name: Read-KairosDB-00**                                      | **#vcores**          | **Mem[GB]** | **Net in[Kb/s]** | **Net out[Kb/s]** |
| Baseline                             |                 0.06 |        24.5 |              613 |              73 |
| One Rack, One Pod, Continuous        |                 0.16 |        24.3 |             2148 |               122 |
| One Rack, One Pod, Discrete          |                 0.11 |        24.3 |             1848 |               108 |
| All Racks, Multiple Pods, Continuous |                  0.4 |        24.4 |             5971 |               295 |
| All Racks, Multiple Pods, Discrete   |                 0.12 |        24.3 |             1638 |               102 |
| All Racks, One Pod, Continuous       |                 0.16 |        24.3 |             1804 |               310 |
| All Racks, One Pod, Discrete         |                  0.1 |        24.3 |             1502 |              99 |
|     **Configuration Name: Write-KairosDB-00**                                   | **#vcores**          | **Mem[GB]** | **Net in[Kb/s]** | **Net out[Kb/s]** |
| Baseline                             |                 1.84 |        23.8 |             3450 |              3739 |
| One Rack, One Pod, Continuous        |                    2 |        23.7 |             3426 |              3784 |
| One Rack, One Pod, Discrete          |                 1.84 |        23.7 |             3244 |              3757 |
| All Racks, Multiple Pods, Continuous |                  2.4 |        23.6 |             2747 |              3866 |
| All Racks, Multiple Pods, Discrete   |                 1.84 |        23.7 |             2871 |              3760 |
| All Racks, One Pod, Continuous       |                 1.92 |        23.7 |             2983 |              3803 |
| All Racks, One Pod, Discrete         |                 1.84 |        23.7 |             3351 |              3780 |
|        **Configuration Name: Read-KairosDB-01**                                 | **#vcores**          | **Mem[GB]** | **Net in[Kb/s]** | **Net out[Kb/s]** |
| Baseline                             |                 0.03 |        24.3 |              529 |                17 |
| One Rack, One Pod, Continuous        |                 0.13 |        24.1 |             2167 |              2148 |
| One Rack, One Pod, Discrete          |                 0.08 |        24.1 |             1941 |                53 |
| All Racks, Multiple Pods, Continuous |                 0.38 |        24.2 |             5809 |               233 |
| All Racks, Multiple Pods, Discrete   |                 0.09 |        24.2 |             1646 |                53 |
| All Racks, One Pod, Continuous       |                 0.12 |        24.1 |             1567 |               247 |
| All Racks, One Pod, Discrete         |                 0.06 |        24.1 |             1428 |                45 |
|   **Configuration Name: Cassandra**                                       | **#vcores**          | **Mem[GB]** | **Net in[Kb/s]** | **Net out[Kb/s]** |
| Baseline                             |                 0.97 |      114.72 |             1240 |              2078 |
| One Rack, One Pod, Continuous        |                 1.54 |         114 |             1756 |              3480 |
| One Rack, One Pod, Discrete          |                 1.37 |      114.72 |             1656 |              3019 |
| All Racks, Multiple Pods, Continuous |                 2.74 |       114.8 |             2501 |              6887 |
| All Racks, Multiple Pods, Discrete   |                 1.45 |      114.64 |             1632 |              3302 |
| All Racks, One Pod, Continuous       |                 3.35 |       114.8 |             1846 |              3199 |
| All Racks, One Pod, Discrete         |                 1.22 |      114.72 |             1481 |              2892 |
|                   **Configuration Name: Total**                   | **#vcores**          | **Mem[GB]** | **Net in[Kb/s]** | **Net out[Kb/s]** |
| Baseline                             |                 3.08 |      189.96 |             6670 |              6739 |
| One Rack, One Pod, Continuous        |                 4.47 |       188.7 |            10457 |             10490 |
| One Rack, One Pod, Discrete          |                 3.59 |      189.41 |             9588 |              7832 |
| All Racks, Multiple Pods, Continuous |                  7.9 |      189.64 |            18310 |             12570 |
| All Racks, Multiple Pods, Discrete   |                 3.84 |      189.45 |             8724 |              8152 |
| All Racks, One Pod, Continuous       |                 5.87 |      189.51 |             9190 |              8546 |
| All Racks, One Pod, Discrete         |                 3.41 |      189.43 |             8686 |              7738 |
