# STAR_FORMATION REPO

This repository contains all notebooks and results written and obtained by B. Crompvoets (MSc candidate University of Victoria, Canada) as part of their MSc thesis. 

### Phase 1
This phase is mostly concerned with data exploration and contains a naive reproduction of the multi-layer perceptron created by Cornu and Montillaud ("A neural network-based approach to YSO classification"; 2021). Several classical methods are also performed and test the validity of the targets obtained by Cornu and Montillaud (2021). A report details the project and it's results within this folder.

### Phase 2
In Phase 2, the classification of all objects available is explored. This phase is a direct result of the misclassification of objects within the first phase, and focuses on determining if added classes can aid in the classification of objects. Although improvements in some respects are noted, this approach has been discarded due to a better methodology being determined.

### Phase 3.1
Phase 3 seeks to create networks for both the IRAC and MIPS instruments on board the Spitzer space telescope. The methodology in this phase involves first separating objects into one of three categories: Young Stellar Objects (YSOs), Extra-Galactic sources (EGs), and Stars. A second network then classifies those objects classified as YSOs into further categories of Class I, Flat-Spectrum, Class II, and Class III YSOs. Although gradient boosting and random forest (the classical methods previously determined in Phase 1 to best separate objects), are included for comparison, they do not perform as well as the MLP. 