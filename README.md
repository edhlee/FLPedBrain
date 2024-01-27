# FLPedBrain

FL-PedBrain: A Federated Learning AI Platform for Pediatric Brain Tumors, an International Study

There are two implementations of the FL algorithm. The first performs client-side training on dedicated devices (GPUs) - 1 client (hospital site) per device. The other performs all client-side training in 1 device. Use the latter if GPU memory can fit all 16 sites' training graphs.   

Note: For our current real-time FL project with hospitals (not simulated), please contact me directly.

More details (with organized data) to come shortly!


## Description

While multiple factors, both biological and external, impact disease, AI studies in medicine are often confined to small and non-diverse patient cohorts. Such limitation typically stems from obstacles of large-scale data sharing and data privacy issues. Federated learning (FL) has emerged as one potential solution for AI developments, enabling training across a network of hospitals without direct data sharing. Here, we present an FL platform for pediatric posterior fossa brain tumors, FL-PedBrain, and evaluate its performance on a diverse and realistic multi-center pediatric cohort. We target pediatric brain tumors given the overall scarcity of such datasets, even within tertiary care pediatric hospitals. Our platform orchestrates federated training that performs an end-to-end joint tumor classification and segmentation across 19 participating international sites. FL-PedBrain exhibits less than a 1.5% decrease in classification and a 3% reduction in segmentation performance compared to the traditional approach using training with centrally shared data. We find that federated training boosts performance compared to a model trained solely on the largest single site. For example, FL boosts segmentation performance from 20 to 30% on three external and out-of-network, hold-out sites. Finally, we explore the underlying sources of data heterogeneity, such as variations in image quality, and examine robustness of FL in real world scenarios due to data imbalances.

<img width="468" alt="image" src="https://github.com/edhlee/FLPedBrain/assets/12375462/0309e543-b9a4-4eb5-9e55-339b5fe74dc5">
